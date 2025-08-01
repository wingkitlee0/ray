import io
import json
import logging
import os
import re
import string
import sys
import time
import uuid
from contextlib import redirect_stderr
from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

import httpx
import pytest
import starlette
from fastapi import FastAPI
from starlette.responses import PlainTextResponse

import ray
import ray.util.state as state_api
from ray import serve
from ray._common.test_utils import wait_for_condition
from ray._private.ray_logging.formatters import JSONFormatter
from ray.serve._private.common import DeploymentID, ReplicaID, ServeComponentType
from ray.serve._private.constants import SERVE_LOG_EXTRA_FIELDS, SERVE_LOGGER_NAME
from ray.serve._private.logging_utils import (
    ServeComponentFilter,
    ServeFormatter,
    StreamToLogger,
    configure_component_logger,
    configure_default_serve_logger,
    get_serve_logs_dir,
    redirected_print,
)
from ray.serve._private.test_utils import get_application_url
from ray.serve._private.utils import get_component_file_name
from ray.serve.context import _get_global_client
from ray.serve.schema import EncodingType, LoggingConfig


class FakeLogger:
    def __init__(self):
        self._logs: List[Tuple[int, str]] = []

    def log(self, level: int, message: str, stacklevel: int = 1):
        self._logs.append((level, message))

    def get_logs(self):
        return self._logs


class FakeStdOut:
    def __init__(self):
        self.encoding = "utf-8"


@pytest.fixture
def serve_and_ray_shutdown():
    serve.shutdown()
    ray.shutdown()
    yield


def set_logging_config(monkeypatch, max_bytes, backup_count):
    monkeypatch.setenv("RAY_ROTATION_MAX_BYTES", str(max_bytes))
    monkeypatch.setenv("RAY_ROTATION_BACKUP_COUNT", str(backup_count))


def _get_expected_replica_log_content(replica_id: ReplicaID):
    app_name = replica_id.deployment_id.app_name
    deployment_name = replica_id.deployment_id.name
    return f"{app_name}_{deployment_name} {replica_id.unique_id}"


def test_log_rotation_config(monkeypatch, ray_shutdown):
    # This test should be executed before any test that uses
    # the shared serve_instance, as environment variables
    # for log rotation need to be set before ray.init
    logger = logging.getLogger("ray.serve")
    max_bytes = 100
    backup_count = 3
    set_logging_config(monkeypatch, max_bytes, backup_count)
    ray.init(num_cpus=1)

    @serve.deployment
    class Handle:
        def __call__(self):
            handlers = logger.handlers
            res = {}
            for handler in handlers:
                if isinstance(handler, logging.handlers.MemoryHandler):
                    target = handler.target
                    assert isinstance(target, logging.handlers.RotatingFileHandler)
                    res["max_bytes"] = target.maxBytes
                    res["backup_count"] = target.backupCount
            return res

    handle = serve.run(Handle.bind())
    rotation_config = handle.remote().result()
    assert rotation_config["max_bytes"] == max_bytes
    assert rotation_config["backup_count"] == backup_count


@pytest.mark.parametrize("log_format", ["TEXT", "JSON"])
def test_http_access_log_in_stderr(serve_instance, log_format):
    if log_format == "JSON":
        # TODO (SERVE-908|harshit): This test is flaky in premerge.
        pytest.skip("The test for JSON log format is flaky, skipping for now.")

    name = "deployment_name"

    fastapi_app = FastAPI()

    @serve.deployment(name=name)
    @serve.ingress(fastapi_app)
    class Handler:
        def __init__(self):
            self._replica_unique_id = serve.get_replica_context().replica_id.unique_id

        @fastapi_app.get("/")
        def get_root(self):
            return PlainTextResponse(self._replica_unique_id)

        @fastapi_app.post("/")
        def post_root(self):
            return PlainTextResponse(self._replica_unique_id)

        @fastapi_app.get("/{status}")
        def template(self, status: str):
            return PlainTextResponse(self._replica_unique_id, status_code=int(status))

        @fastapi_app.put("/fail")
        def fail(self):
            raise RuntimeError("OOPS!")

    serve.run(Handler.bind(), logging_config={"encoding": log_format})

    f = io.StringIO()
    with redirect_stderr(f):

        def check_log(
            replica_id: ReplicaID,
            method: str,
            route: str,
            status_code: str,
            fail: bool = False,
        ):
            s = f.getvalue()
            return all(
                [
                    name in s,
                    _get_expected_replica_log_content(replica_id) in s,
                    f"-- {method} {route} {status_code}" in s,
                    "ms" in s,
                    ("OOPS!" in s and "RuntimeError" in s)
                    if fail
                    else True,  # Check for stacktrace.
                ]
            )

        url = get_application_url(use_localhost=True)

        r = httpx.get(url)
        assert r.status_code == 200
        replica_id = ReplicaID(unique_id=r.text, deployment_id=DeploymentID(name=name))
        wait_for_condition(
            check_log,
            replica_id=replica_id,
            method="GET",
            route="/",
            status_code="200",
            timeout=20,
        )

        r = httpx.post(url)
        assert r.status_code == 200
        wait_for_condition(
            check_log,
            replica_id=replica_id,
            method="POST",
            route="/",
            status_code="200",
            timeout=20,
        )

        r = httpx.get(f"{url}/350")
        assert r.status_code == 350
        wait_for_condition(
            check_log,
            replica_id=replica_id,
            method="GET",
            route="/{status}",
            status_code="350",
            timeout=20,
        )

        r = httpx.put(f"{url}/fail")
        assert r.status_code == 500
        wait_for_condition(
            check_log,
            replica_id=replica_id,
            method="PUT",
            route="/fail",
            status_code="500",
            fail=True,
            timeout=20,
        )


@pytest.mark.parametrize("log_format", ["TEXT", "JSON"])
def test_http_access_log_in_logs_file(serve_instance, log_format):
    name = "deployment_name"
    fastapi_app = FastAPI()

    @serve.deployment(name=name)
    @serve.ingress(fastapi_app)
    class Handler:
        def __init__(self):
            self._replica_unique_id = serve.get_replica_context().replica_id.unique_id

        def _get_context_info(self):
            """Get context information for matching with logs"""
            request_context = ray.serve.context._get_serve_request_context()
            return {
                "replica": self._replica_unique_id,
                "request_id": request_context.request_id,
                "worker_id": ray.get_runtime_context().get_worker_id(),
                "node_id": ray.get_runtime_context().get_node_id(),
                "actor_id": ray.get_runtime_context().get_actor_id(),
            }

        @fastapi_app.get("/")
        def get_root(self):
            return self._get_context_info()

        @fastapi_app.post("/")
        def post_root(self):
            return self._get_context_info()

        @fastapi_app.get("/{status}")
        def template(self, status: str):
            content_info = {"context": self._get_context_info()}
            return PlainTextResponse(
                content=json.dumps(content_info),
                status_code=int(status),
                media_type="application/json",
            )

        @fastapi_app.put("/fail")
        def fail(self):
            error_response = {"error": "OOPS!", "context": self._get_context_info()}
            return PlainTextResponse(
                content=json.dumps(error_response),
                status_code=500,
                media_type="application/json",
            )

    serve.run(Handler.bind(), logging_config={"encoding": log_format})

    # Get log file information
    client = _get_global_client()
    serve_log_dir = get_serve_logs_dir()
    replicas = ray.get(
        client._controller.get_deployment_details.remote("default", name)
    ).replicas
    replica_id = replicas[0].replica_id
    replica_log_file_name = f"replica_default_{name}_{replica_id}.log"
    log_file_path = os.path.join(serve_log_dir, replica_log_file_name)

    url = get_application_url(use_localhost=True)

    # Define the HTTP calls to make
    http_calls = [
        {
            "method": "GET",
            "url": url,
            "expected_status": 200,
            "expected_route": "/",
        },
        {
            "method": "POST",
            "url": url,
            "expected_status": 200,
            "expected_route": "/",
        },
        {
            "method": "GET",
            "url": f"{url}/350",
            "expected_status": 350,
            "expected_route": "/{status}",
        },
        {
            "method": "PUT",
            "url": f"{url}/fail",
            "expected_status": 500,
            "expected_route": "/fail",
        },
    ]

    def get_file_end_position(file_path):
        """Get the current end position of the file"""
        try:
            with open(file_path, "r") as f:
                f.seek(0, 2)  # Seek to end of file
                return f.tell()
        except FileNotFoundError:
            return 0

    def verify_http_response_in_logs(
        response, new_log_lines, call_info, log_format, context_info=None
    ):
        """Verify that the HTTP response matches the new log entries"""
        if not new_log_lines:
            print("No new log lines found")
            return False

        if log_format == "JSON":
            for line in new_log_lines:
                if line.strip():
                    try:
                        log_data = json.loads(line.strip())
                        message = log_data.get("message", "")

                        if all(
                            [
                                f"default_{name}" == log_data.get("deployment"),
                                f"{call_info['method']} {call_info['expected_route']} {call_info['expected_status']}"
                                in message,
                                "ms" in message,
                                (
                                    context_info is not None
                                    and log_data.get("request_id")
                                    == context_info["request_id"]
                                    and log_data.get("worker_id")
                                    == context_info["worker_id"]
                                    and log_data.get("node_id")
                                    == context_info["node_id"]
                                    and log_data.get("replica")
                                    == context_info["replica"]
                                ),
                            ]
                        ):
                            return True

                    except json.JSONDecodeError:
                        continue
        else:
            for line in new_log_lines:
                if all(
                    [
                        name in line,
                        f"default_{name} {replica_id}" in line,
                        f"-- {call_info['method']} {call_info['expected_route']} {call_info['expected_status']}"
                        in line,
                        "ms" in line,
                    ]
                ):
                    return True

        return False

    # Process each HTTP call individually
    for i, call_info in enumerate(http_calls):
        # Step 1: Get current file end position
        start_position = get_file_end_position(log_file_path)

        # Step 2: Make HTTP call
        if call_info["method"] == "GET":
            response = httpx.get(call_info["url"])
        elif call_info["method"] == "POST":
            response = httpx.post(call_info["url"])
        elif call_info["method"] == "PUT":
            response = httpx.put(call_info["url"])
        else:
            raise ValueError(f"Unsupported HTTP method: {call_info['method']}")

        # Verify response status
        assert (
            response.status_code == call_info["expected_status"]
        ), f"Expected status {call_info['expected_status']}, got {response.status_code}"

        # Extract context information from response
        context_info = None
        response_data = response.json()

        # For all routes apart from `/` endpoint, context info is nested under "context" key
        if call_info["expected_route"] == "/":
            context_info = response_data
        elif "context" in response_data:
            context_info = response_data["context"]
        else:
            raise ValueError(
                f"Could not extract context info from response: {response.text}"
            )

        # Step 3: Verify HTTP response matches new log lines
        def verify_log_lines(
            file_path, start_pos, response, call_info, log_format, context_info
        ):
            new_log_lines = []
            try:
                with open(file_path, "r") as f:
                    f.seek(start_pos)
                    new_content = f.read()
                    lines = new_content.splitlines() if new_content else []
                    new_log_lines = lines
            except FileNotFoundError:
                new_log_lines = []

            return verify_http_response_in_logs(
                response, new_log_lines, call_info, log_format, context_info
            )

        wait_for_condition(
            verify_log_lines,
            timeout=20,
            retry_interval_ms=100,
            file_path=log_file_path,
            start_pos=start_position,
            response=response,
            call_info=call_info,
            log_format=log_format,
            context_info=context_info,
        )


def test_http_access_log_in_proxy_logs_file(serve_instance):
    name = "deployment_name"
    fastapi_app = FastAPI()

    @serve.deployment(name=name)
    @serve.ingress(fastapi_app)
    class Handler:
        @fastapi_app.get("/")
        def get_root(self):
            return "Hello World!"

    serve.run(Handler.bind(), logging_config={"encoding": "TEXT"})

    # Get log file information
    nodes = state_api.list_nodes()
    serve_log_dir = get_serve_logs_dir()
    node_ip_address = nodes[0].node_ip
    proxy_log_file_name = get_component_file_name(
        "proxy", node_ip_address, component_type=None, suffix=".log"
    )
    proxy_log_path = os.path.join(serve_log_dir, proxy_log_file_name)

    request_id = str(uuid.uuid4())
    response = httpx.get("http://localhost:8000", headers={"X-Request-ID": request_id})
    assert response.status_code == 200

    def verify_request_id_in_logs(proxy_log_path, request_id):
        with open(proxy_log_path, "r") as f:
            for line in f:
                if request_id in line:
                    return True
        return False

    wait_for_condition(
        verify_request_id_in_logs, proxy_log_path=proxy_log_path, request_id=request_id
    )


def test_handle_access_log(serve_instance):
    name = "handler"

    @serve.deployment(name=name)
    class Handler:
        def other_method(self, *args):
            return serve.get_replica_context().replica_id

        def __call__(self, *args):
            return serve.get_replica_context().replica_id

        def throw(self, *args):
            raise RuntimeError("blah blah blah")

    h = serve.run(Handler.bind())

    f = io.StringIO()
    with redirect_stderr(f):

        def check_log(replica_id: ReplicaID, method_name: str, fail: bool = False):
            s = f.getvalue()
            return all(
                [
                    name in s,
                    _get_expected_replica_log_content(replica_id) in s,
                    method_name in s,
                    ("ERROR" if fail else "OK") in s,
                    "ms" in s,
                    ("blah blah blah" in s and "RuntimeError" in s)
                    if fail
                    else True,  # Check for stacktrace.
                ]
            )

        replica_id = h.remote().result()
        wait_for_condition(check_log, replica_id=replica_id, method_name="__call__")

        h.other_method.remote().result()
        wait_for_condition(check_log, replica_id=replica_id, method_name="other_method")

        with pytest.raises(RuntimeError, match="blah blah blah"):
            h.throw.remote().result()

        wait_for_condition(
            check_log, replica_id=replica_id, method_name="throw", fail=True
        )


def test_user_logs(serve_instance):
    logger = logging.getLogger("ray.serve")
    stderr_msg = "user log message"
    log_file_msg = "in file only"
    name = "user_fn"

    @serve.deployment(name=name)
    def fn(*args):
        logger.info(stderr_msg)
        logger.info(log_file_msg, extra={"log_to_stderr": False})
        return (
            serve.get_replica_context().replica_id,
            logger.handlers[1].target.baseFilename,
        )

    handle = serve.run(fn.bind())

    f = io.StringIO()
    with redirect_stderr(f):
        replica_id, log_file_name = handle.remote().result()

        def check_stderr_log(replica_id: ReplicaID):
            s = f.getvalue()
            return all(
                [
                    name in s,
                    _get_expected_replica_log_content(replica_id) in s,
                    stderr_msg in s,
                    log_file_msg not in s,
                ]
            )

        # Only the stderr_msg should be logged to stderr.
        wait_for_condition(check_stderr_log, replica_id=replica_id)

        def check_log_file(replica_id: str):
            with open(log_file_name, "r") as f:
                s = f.read()
                return all(
                    [
                        name in s,
                        _get_expected_replica_log_content(replica_id) in s,
                        stderr_msg in s,
                        log_file_msg in s,
                    ]
                )

        # Both messages should be logged to the file.
        wait_for_condition(check_log_file, replica_id=replica_id)


def test_disable_access_log(serve_instance):
    logger = logging.getLogger("ray.serve")

    @serve.deployment
    class A:
        def __init__(self):
            logger.setLevel(logging.ERROR)

        def __call__(self, *args):
            return serve.get_replica_context().replica_id

    handle = serve.run(A.bind())

    f = io.StringIO()
    with redirect_stderr(f):
        replica_id = handle.remote().result()

        for _ in range(10):
            time.sleep(0.1)
            assert _get_expected_replica_log_content(replica_id) not in f.getvalue()


def test_log_filenames_contain_only_posix_characters(serve_instance):
    """Assert that all log filenames only consist of POSIX-compliant characters.

    See: https://github.com/ray-project/ray/issues/41615
    """

    @serve.deployment
    class A:
        def __call__(self, *args) -> str:
            return "hi"

    serve.run(A.bind())

    url = get_application_url(use_localhost=True)
    r = httpx.get(url)
    r.raise_for_status()
    assert r.text == "hi"

    acceptable_chars = string.ascii_letters + string.digits + "_" + "."
    for filename in os.listdir(get_serve_logs_dir()):
        assert all(char in acceptable_chars for char in filename)


@pytest.mark.parametrize("json_log_format", [False, True])
def test_context_information_in_logging(serve_and_ray_shutdown, json_log_format):
    """Make sure all context information exist in the log message"""

    logger = logging.getLogger("ray.serve")

    @serve.deployment(
        logging_config={"encoding": "JSON" if json_log_format else "TEXT"}
    )
    def fn(*args):
        logger.info("user func")
        request_context = ray.serve.context._get_serve_request_context()
        return {
            "request_id": request_context.request_id,
            "route": request_context.route,
            "app_name": request_context.app_name,
            "log_file": logger.handlers[1].target.baseFilename,
            "replica": serve.get_replica_context().replica_id.unique_id,
            "actor_id": ray.get_runtime_context().get_actor_id(),
            "worker_id": ray.get_runtime_context().get_worker_id(),
            "node_id": ray.get_runtime_context().get_node_id(),
            "task_name": ray.get_runtime_context().get_task_name(),
            "task_func_name": ray.get_runtime_context().get_task_function_name(),
            "actor_name": ray.get_runtime_context().get_actor_name(),
        }

    @serve.deployment(
        logging_config={"encoding": "JSON" if json_log_format else "TEXT"}
    )
    class Model:
        def __call__(self, req: starlette.requests.Request):
            logger.info("user log message from class method")
            request_context = ray.serve.context._get_serve_request_context()
            return {
                "request_id": request_context.request_id,
                "route": request_context.route,
                "app_name": request_context.app_name,
                "log_file": logger.handlers[1].target.baseFilename,
                "replica": serve.get_replica_context().replica_id.unique_id,
                "actor_id": ray.get_runtime_context().get_actor_id(),
                "worker_id": ray.get_runtime_context().get_worker_id(),
                "node_id": ray.get_runtime_context().get_node_id(),
                "task_name": ray.get_runtime_context().get_task_name(),
                "task_func_name": ray.get_runtime_context().get_task_function_name(),
                "actor_name": ray.get_runtime_context().get_actor_name(),
            }

    serve.run(fn.bind(), name="app1", route_prefix="/fn")
    serve.run(Model.bind(), name="app2", route_prefix="/class_method")

    url = get_application_url(app_name="app1", use_localhost=True)
    url2 = get_application_url(app_name="app2", use_localhost=True)

    f = io.StringIO()
    with redirect_stderr(f):
        resp = httpx.get(url).json()
        resp2 = httpx.get(url2).json()

        # Check the component log
        expected_log_infos = [
            f"{resp['request_id']} -- ",
            f"{resp2['request_id']} -- ",
        ]

        # Check User log
        user_log_regexes = [
            f".*{resp['request_id']} -- user func.*",
            f".*{resp2['request_id']} -- user log.*message from class method.*",
        ]

        def check_log():
            logs_content = f.getvalue()
            for expected_log_info in expected_log_infos:
                assert expected_log_info in logs_content
            for regex in user_log_regexes:
                assert re.findall(regex, logs_content) != []
            return True

        # Check stream log
        wait_for_condition(
            check_log,
            timeout=25,
            retry_interval_ms=100,
        )

        # Check user log file
        method_replica_id = resp["replica"].split("#")[-1]
        class_method_replica_id = resp2["replica"].split("#")[-1]
        if json_log_format:
            user_method_log_regex = (
                '.*"message": "user func".*'
                f'"route": "{resp["route"]}", '
                f'"request_id": "{resp["request_id"]}", '
                f'"application": "{resp["app_name"]}", '
                f'"worker_id": "{resp["worker_id"]}", '
                f'"node_id": "{resp["node_id"]}", '
                f'"actor_id": "{resp["actor_id"]}", '
                f'"task_name": "{resp["task_name"]}", '
                f'"task_func_name": "{resp["task_func_name"]}", '
                f'"actor_name": "{resp["actor_name"]}", '
                f'"deployment": "{resp["app_name"]}_fn", '
                f'"replica": "{method_replica_id}", '
                f'"component_name": "replica", '
                rf'"timestamp_ns": \d+}}.*'
            )
            user_class_method_log_regex = (
                '.*"message": "user log message from class method".*'
                f'"route": "{resp2["route"]}", '
                f'"request_id": "{resp2["request_id"]}", '
                f'"application": "{resp2["app_name"]}", '
                f'"worker_id": "{resp2["worker_id"]}", '
                f'"node_id": "{resp2["node_id"]}", '
                f'"actor_id": "{resp2["actor_id"]}", '
                f'"task_name": "{resp2["task_name"]}", '
                f'"task_func_name": "{resp2["task_func_name"]}", '
                f'"actor_name": "{resp2["actor_name"]}", '
                f'"deployment": "{resp2["app_name"]}_Model", '
                f'"replica": "{class_method_replica_id}", '
                f'"component_name": "replica", '
                rf'"timestamp_ns": \d+}}.*'
            )
        else:
            user_method_log_regex = f".*{resp['request_id']} -- user func.*"
            user_class_method_log_regex = (
                f".*{resp2['request_id']} -- .*user log message from class method.*"
            )

        def check_log_file(log_file: str, expected_regex: list):
            with open(log_file, "r") as f:
                s = f.read()
                assert re.findall(expected_regex, s) != []

        check_log_file(resp["log_file"], user_method_log_regex)
        check_log_file(resp2["log_file"], user_class_method_log_regex)


@pytest.mark.parametrize("raise_error", [True, False])
def test_extra_field(serve_and_ray_shutdown, raise_error):
    """Test ray serve extra logging"""
    logger = logging.getLogger("ray.serve")

    @serve.deployment(logging_config={"encoding": "JSON"})
    def fn(*args):
        if raise_error:
            logger.info("user_func", extra={SERVE_LOG_EXTRA_FIELDS: [123]})
        else:
            logger.info(
                "user_func",
                extra={"k1": "my_v1", SERVE_LOG_EXTRA_FIELDS: {"k2": "my_v2"}},
            )
        return {
            "log_file": logger.handlers[1].target.baseFilename,
        }

    serve.run(fn.bind(), name="app1", route_prefix="/fn")
    url = get_application_url(app_name="app1", use_localhost=True)

    resp = httpx.get(url)
    if raise_error:
        resp.status_code == 500
    else:
        resp = resp.json()
        with open(resp["log_file"], "r") as f:
            s = f.read()
            assert re.findall(".*my_v1.*", s) != []
            assert re.findall('.*"k2": "my_v2".*', s) != []


def check_log_file(log_file: str, expected_regex: list, check_contains: bool = True):
    with open(log_file, "r") as f:
        s = f.read()
        for regex in expected_regex:
            if check_contains:
                assert re.findall(regex, s) != []
            else:
                assert re.findall(regex, s) == []


class TestLoggingAPI:
    def test_start_serve_with_logging_config(self, serve_and_ray_shutdown):
        serve.start(logging_config={"log_level": "DEBUG", "encoding": "JSON"})
        serve_log_dir = get_serve_logs_dir()
        # Check controller log
        actors = state_api.list_actors()
        expected_log_regex = [".*logger with logging config.*"]
        for actor in actors:
            print(actor["name"])
            if "SERVE_CONTROLLER_ACTOR" == actor["name"]:
                controller_pid = actor["pid"]
        controller_log_file_name = get_component_file_name(
            "controller", controller_pid, component_type=None, suffix=".log"
        )
        controller_log_path = os.path.join(serve_log_dir, controller_log_file_name)
        check_log_file(controller_log_path, expected_log_regex)

        # Check proxy log
        nodes = state_api.list_nodes()
        node_ip_address = nodes[0].node_ip
        proxy_log_file_name = get_component_file_name(
            "proxy", node_ip_address, component_type=None, suffix=".log"
        )
        proxy_log_path = os.path.join(serve_log_dir, proxy_log_file_name)
        check_log_file(proxy_log_path, expected_log_regex)

    @pytest.mark.parametrize("encoding_type", ["TEXT", "JSON"])
    def test_encoding(self, serve_and_ray_shutdown, encoding_type):
        """Test serve.run logging API"""
        logging_config = {"encoding": encoding_type}
        logger = logging.getLogger("ray.serve")

        @serve.deployment(logging_config=logging_config)
        class Model:
            def __call__(self, req: starlette.requests.Request):
                return {
                    "log_file": logger.handlers[1].target.baseFilename,
                    "replica": serve.get_replica_context().replica_id.unique_id,
                }

        serve.run(Model.bind())
        url = get_application_url(use_localhost=True)

        resp = httpx.get(url).json()

        replica_id = resp["replica"].split("#")[-1]
        if encoding_type == "JSON":
            expected_log_regex = [f'"replica": "{replica_id}", ']
        else:
            expected_log_regex = [f".*{replica_id}.*"]
        check_log_file(resp["log_file"], expected_log_regex)

    def test_log_level(self, serve_and_ray_shutdown):
        logger = logging.getLogger("ray.serve")

        @serve.deployment
        class Model:
            def __call__(self, req: starlette.requests.Request):
                logger.info("model_info_level")
                logger.debug("model_debug_level")
                return {
                    "log_file": logger.handlers[1].target.baseFilename,
                }

        serve.run(Model.bind())
        url = get_application_url(use_localhost=True)

        resp = httpx.get(url).json()
        expected_log_regex = [".*model_info_level.*"]
        check_log_file(resp["log_file"], expected_log_regex)

        # Make sure 'model_debug_level' log content does not exist
        with pytest.raises(AssertionError):
            check_log_file(resp["log_file"], [".*model_debug_level.*"])

        serve.run(Model.options(logging_config={"log_level": "DEBUG"}).bind())
        url = get_application_url(use_localhost=True)

        resp = httpx.get(url).json()
        expected_log_regex = [".*model_info_level.*", ".*model_debug_level.*"]
        check_log_file(resp["log_file"], expected_log_regex)

    def test_logs_dir(self, serve_and_ray_shutdown):
        logger = logging.getLogger("ray.serve")

        @serve.deployment
        class Model:
            def __call__(self, req: starlette.requests.Request):
                logger.info("model_info_level")
                for handler in logger.handlers:
                    if isinstance(handler, logging.handlers.MemoryHandler):
                        target = handler.target
                        assert isinstance(target, logging.handlers.RotatingFileHandler)
                        return {
                            "logs_path": target.baseFilename,
                        }
                raise AssertionError("No memory handler found")

        serve.run(Model.bind())
        url = get_application_url(use_localhost=True)

        resp = httpx.get(url).json()

        paths = resp["logs_path"].split("/")
        paths[-1] = "new_dir"
        new_log_dir = "/".join(paths)

        serve.run(
            Model.options(
                logging_config={
                    "logs_dir": new_log_dir,
                    "additional_log_standard_attrs": ["name"],
                }
            ).bind()
        )
        url = get_application_url(use_localhost=True)

        resp = httpx.get(url).json()
        assert "new_dir" in resp["logs_path"]

        check_log_file(resp["logs_path"], [".*model_info_level.*"])
        check_log_file(resp["logs_path"], ["ray.serve"], check_contains=True)

    @pytest.mark.parametrize("enable_access_log", [True, False])
    @pytest.mark.parametrize("encoding_type", ["TEXT", "JSON"])
    def test_access_log(self, serve_and_ray_shutdown, encoding_type, enable_access_log):
        logger = logging.getLogger("ray.serve")
        logging_config = {
            "enable_access_log": enable_access_log,
            "encoding": encoding_type,
        }

        @serve.deployment(logging_config=logging_config)
        class Model:
            def __call__(self, req: starlette.requests.Request):
                logger.info("model_info_level")
                logger.info("model_not_show", extra={"serve_access_log": True})
                return {
                    "logs_path": logger.handlers[1].target.baseFilename,
                }

        serve.run(Model.bind())
        url = get_application_url(use_localhost=True)

        resp = httpx.get(url)
        assert resp.status_code == 200
        resp = resp.json()
        check_log_file(resp["logs_path"], [".*model_info_level.*"])
        if enable_access_log:
            check_log_file(resp["logs_path"], [".*model_not_show.*"])
            check_log_file(
                resp["logs_path"], ["serve_access_log"], check_contains=False
            )
        else:
            with pytest.raises(AssertionError):
                check_log_file(resp["logs_path"], [".*model_not_show.*"])

    @pytest.mark.parametrize("encoding_type", ["TEXT", "JSON"])
    def test_additional_log_standard_attrs(self, serve_and_ray_shutdown, encoding_type):
        """Test additional log standard attrs"""
        logger = logging.getLogger("ray.serve")
        logging_config = {
            "enable_access_log": True,
            "encoding": encoding_type,
            "additional_log_standard_attrs": ["name"],
        }

        @serve.deployment(logging_config=logging_config)
        class Model:
            def __call__(self, req: starlette.requests.Request):
                logger.info("model_info_level")
                logger.info("model_not_show", extra={"serve_access_log": True})
                return {
                    "logs_path": logger.handlers[1].target.baseFilename,
                }

        serve.run(Model.bind())
        url = get_application_url(use_localhost=True)

        resp = httpx.get(url)
        assert resp.status_code == 200
        resp = resp.json()
        if encoding_type == "JSON":
            check_log_file(resp["logs_path"], ["name"], check_contains=True)
        else:
            check_log_file(resp["logs_path"], ["ray.serve"], check_contains=True)

    def test_application_logging_overwrite(self, serve_and_ray_shutdown):
        @serve.deployment
        class Model:
            def __call__(self, req: starlette.requests.Request):
                logger = logging.getLogger("ray.serve")
                logger.info("model_info_level")
                logger.debug("model_debug_level")
                return {
                    "log_file": logger.handlers[1].target.baseFilename,
                }

        serve.run(Model.bind(), logging_config={"log_level": "DEBUG"})
        url = get_application_url(use_localhost=True)

        resp = httpx.get(url).json()
        expected_log_regex = [".*model_info_level.*", ".*model_debug_level.*"]
        check_log_file(resp["log_file"], expected_log_regex)

        # Setting logging config in the deployment level, application
        # config can't override it.

        @serve.deployment(logging_config={"log_level": "INFO"})
        class Model2:
            def __call__(self, req: starlette.requests.Request):
                logger = logging.getLogger("ray.serve")
                logger.info("model_info_level")
                logger.debug("model_debug_level")
                return {
                    "log_file": logger.handlers[1].target.baseFilename,
                }

        serve.run(
            Model2.bind(),
            logging_config={"log_level": "DEBUG"},
            name="app2",
            route_prefix="/app2",
        )
        url = get_application_url(app_name="app2", use_localhost=True)

        resp = httpx.get(url).json()
        check_log_file(resp["log_file"], [".*model_info_level.*"])
        # Make sure 'model_debug_level' log content does not exist.
        with pytest.raises(AssertionError):
            check_log_file(resp["log_file"], [".*model_debug_level.*"])


@pytest.mark.parametrize("is_replica_type_component", [False, True])
def test_serve_component_filter(is_replica_type_component):
    """Test Serve component filter"""

    if is_replica_type_component:
        component_type = ServeComponentType.REPLICA
        filter = ServeComponentFilter("component", "component_id", component_type)
    else:
        filter = ServeComponentFilter("component", "component_id")
    init_kwargs = {
        "name": "test_log",
        "level": logging.DEBUG,
        "pathname": "my_path",
        "lineno": 1,
        "msg": "my_message",
        "args": (),
        "exc_info": None,
    }
    record = logging.LogRecord(**init_kwargs)

    def format_and_verify_json_output(record, expected_record: dict):
        filter.filter(record)
        formatted_record_dict = record.__dict__
        for key in expected_record:
            assert key in formatted_record_dict
            assert formatted_record_dict[key] == expected_record[key]

    expected_json = {}
    if is_replica_type_component:
        expected_json["deployment"] = "component"
        expected_json["replica"] = "component_id"
        expected_json["component_name"] = "replica"
    else:
        expected_json["component_name"] = "component"
        expected_json["component_id"] = "component_id"

    # Ensure message exists in the output.
    # Note that there is no "message" key in the record dict until it has been
    # formatted. This check should go before other fields are set and checked.
    expected_json["msg"] = "my_message"
    format_and_verify_json_output(record, expected_json)

    # Set request id
    record.request_id = "request_id"
    expected_json["request_id"] = "request_id"
    format_and_verify_json_output(record, expected_json)

    # Set route
    record.route = "route"
    expected_json["route"] = "route"
    format_and_verify_json_output(record, expected_json)

    # set application
    record.application = "application"
    expected_json["application"] = "application"
    format_and_verify_json_output(record, expected_json)


@pytest.mark.parametrize(
    "log_encoding",
    [
        [None, None, "TEXT"],
        [None, "TEXT", "TEXT"],
        [None, "JSON", "JSON"],
        ["TEXT", None, "TEXT"],
        ["TEXT", "TEXT", "TEXT"],
        ["TEXT", "JSON", "JSON"],
        ["JSON", None, "JSON"],
        ["JSON", "TEXT", "TEXT"],
        ["JSON", "JSON", "JSON"],
        ["FOOBAR", None, "TEXT"],
        ["FOOBAR", "TEXT", "TEXT"],
        ["FOOBAR", "JSON", "JSON"],
    ],
)
def test_configure_component_logger_with_log_encoding_env_text(log_encoding):
    """Test the configure_component_logger function with different log encoding env.

    When the log encoding env is not set, set to "TEXT" or set to unknon values,
    the ServeFormatter should be used. When the log encoding env is set to "JSON",
    the JSONFormatter should be used. Also, the log config should take the
    precedence it's set.
    """
    env_encoding, log_config_encoding, expected_encoding = log_encoding

    with patch("ray.serve.schema.RAY_SERVE_LOG_ENCODING", env_encoding):
        # Clean up logger handlers
        logger = logging.getLogger(SERVE_LOGGER_NAME)
        logger.handlers.clear()

        # Ensure there is no logger handlers before calling configure_component_logger
        assert logger.handlers == []

        if log_config_encoding is None:
            logging_config = LoggingConfig(logs_dir="/tmp/fake_logs_dir")
        else:
            logging_config = LoggingConfig(
                encoding=log_config_encoding, logs_dir="/tmp/fake_logs_dir"
            )
        configure_component_logger(
            component_name="fake_component_name",
            component_id="fake_component_id",
            logging_config=logging_config,
            component_type=ServeComponentType.REPLICA,
            max_bytes=100,
            backup_count=3,
        )

        for handler in logger.handlers:
            if isinstance(handler, logging.handlers.MemoryHandler):
                if expected_encoding == EncodingType.JSON:
                    assert isinstance(handler.target.formatter, JSONFormatter)
                else:
                    assert isinstance(handler.target.formatter, ServeFormatter)

        # Clean up logger handlers
        logger.handlers.clear()


@pytest.mark.skipif(sys.platform == "win32", reason="Fail to create temp dir.")
@pytest.mark.parametrize(
    "ray_instance",
    [
        {"RAY_SERVE_LOG_TO_STDERR": "0"},
    ],
    indirect=True,
)
def test_logging_disable_stdout(serve_and_ray_shutdown, ray_instance, tmp_dir):
    """Test logging when RAY_SERVE_LOG_TO_STDERR is set.

    When RAY_SERVE_LOG_TO_STDERR=0 is set, serve should redirect stdout and stderr to
    serve logger.
    """
    logs_dir = Path(tmp_dir)
    logging_config = LoggingConfig(encoding="JSON", logs_dir=str(logs_dir))
    serve_logger = logging.getLogger("ray.serve")

    @serve.deployment(logging_config=logging_config)
    def disable_stdout():
        serve_logger.info("from_serve_logger")
        print("from_print")
        sys.stdout.write("direct_from_stdout\n")
        sys.stderr.write("direct_from_stderr\n")
        print("this\nis\nmultiline\nlog\n")
        raise RuntimeError("from_error")

    app = disable_stdout.bind()
    serve.run(app)
    url = get_application_url(use_localhost=True)

    httpx.get(url, timeout=None)

    # Check if each of the logs exist in Serve's log files.
    from_serve_logger_check = False
    from_print_check = False
    from_error_check = False
    direct_from_stdout = False
    direct_from_stderr = False
    multiline_log = False
    for log_file in os.listdir(logs_dir):
        if log_file.startswith("replica_default_disable_stdout"):
            with open(logs_dir / log_file) as f:
                for line in f:
                    structured_log = json.loads(line)
                    message = structured_log["message"]
                    exc_text = structured_log.get("exc_text", "")
                    if "from_serve_logger" in message:
                        from_serve_logger_check = True
                    elif "from_print" in message:
                        from_print_check = True

                    # Error was logged from replica directly.
                    elif "from_error" in exc_text:
                        from_error_check = True
                    elif "direct_from_stdout" in message:
                        direct_from_stdout = True
                    elif "direct_from_stderr" in message:
                        direct_from_stderr = True
                    elif "this\nis\nmultiline\nlog\n" in message:
                        multiline_log = True
    assert from_serve_logger_check
    assert from_print_check
    assert from_error_check
    assert direct_from_stdout
    assert direct_from_stderr
    assert multiline_log


@pytest.mark.skipif(sys.platform == "win32", reason="Fail to look for temp dir.")
def test_serve_logging_file_names(serve_and_ray_shutdown, ray_instance):
    """Test to ensure the log file names are correct."""
    logs_dir = Path("/tmp/ray/session_latest/logs/serve")
    logging_config = LoggingConfig(encoding="JSON")

    @serve.deployment
    def app():
        return "foo"

    app = app.bind()
    serve.run(app, logging_config=logging_config)
    url = get_application_url(use_localhost=True)

    r = httpx.get(url)
    assert r.status_code == 200

    # Construct serve log file names.
    client = _get_global_client()
    controller_id = ray.get(client._controller.get_pid.remote())
    proxy_id = ray.util.get_node_ip_address()
    replicas = ray.get(
        client._controller.get_deployment_details.remote("default", "app")
    ).replicas
    replica_id = replicas[0].replica_id
    controller_log_file_name = f"controller_{controller_id}.log"
    proxy_log_file_name = f"proxy_{proxy_id}.log"
    replica_log_file_name = f"replica_default_app_{replica_id}.log"

    # Check if each of the log files exist.
    controller_log_file_name_correct = False
    proxy_log_file_name_correct = False
    replica_log_file_name_correct = False
    for log_file in os.listdir(logs_dir):
        if log_file == controller_log_file_name:
            controller_log_file_name_correct = True
        elif log_file == proxy_log_file_name:
            proxy_log_file_name_correct = True
        elif log_file == replica_log_file_name:
            replica_log_file_name_correct = True

    assert controller_log_file_name_correct
    assert proxy_log_file_name_correct
    assert replica_log_file_name_correct


def test_stream_to_logger():
    """Test calling methods on StreamToLogger."""
    logger = FakeLogger()
    stdout_object = FakeStdOut()
    stream_to_logger = StreamToLogger(logger, logging.INFO, stdout_object)
    assert logger.get_logs() == []

    # Calling isatty() should return True.
    assert stream_to_logger.isatty() is True

    # Logs are buffered and not flushed to logger.
    stream_to_logger.write("foo")
    assert logger.get_logs() == []

    # Logs are flushed when the message ends with newline "\n".
    stream_to_logger.write("bar\n")
    assert logger.get_logs() == [(20, "foobar")]

    # Calling flush directly can also flush the message to the logger.
    stream_to_logger.write("baz")
    assert logger.get_logs() == [(20, "foobar")]
    stream_to_logger.flush()
    assert logger.get_logs() == [(20, "foobar"), (20, "baz")]

    # Calling the attribute on the StreamToLogger should return the attribute on
    # the stdout object.
    assert stream_to_logger.encoding == stdout_object.encoding

    # Calling non-existing attribute on the StreamToLogger should still raise error.
    with pytest.raises(AttributeError):
        _ = stream_to_logger.i_dont_exist


@pytest.mark.skipif(sys.platform == "win32", reason="Fail to create temp dir.")
@pytest.mark.parametrize(
    "ray_instance",
    [
        {"RAY_SERVE_LOG_TO_STDERR": "0"},
    ],
    indirect=True,
)
def test_json_logging_with_unpickleable_exc_info(
    serve_and_ray_shutdown, ray_instance, tmp_dir
):
    """Test the json logging with unpickleable exc_info.

    exc_info field is often used to log the exception stack trace. However, we had issue
    where deepcopy is applied to traceback object from exc_info which is not pickleable
    and caused logging error.

    See: https://github.com/ray-project/ray/issues/45912
    """
    logs_dir = Path(tmp_dir)
    logging_config = LoggingConfig(encoding="JSON", logs_dir=str(logs_dir))
    logger = logging.getLogger("ray.serve")

    @serve.deployment(logging_config=logging_config)
    class App:
        def __call__(self):
            try:
                raise Exception("fake_exception")
            except Exception as e:
                logger.info("log message", exc_info=e)
            return "foo"

    serve.run(App.bind())
    url = get_application_url(use_localhost=True)

    r = httpx.get(f"{url}")
    assert r.status_code == 200
    for log_file in os.listdir(logs_dir):
        with open(logs_dir / log_file) as f:
            assert "Logging error" not in f.read()
            assert "cannot pickle" not in f.read()


@pytest.mark.skipif(sys.platform == "win32", reason="Fail to create temp dir.")
@pytest.mark.parametrize(
    "ray_instance",
    [
        {"RAY_SERVE_LOG_TO_STDERR": "0"},
    ],
    indirect=True,
)
def test_configure_default_serve_logger_with_stderr_redirect(
    serve_and_ray_shutdown, ray_instance, tmp_dir
):
    """Test configuring default serve logger with stderr redirect.

    Default serve logger should only be configured with one StreamToLogger handler, and
    print, stdout, and stderr should NOT be overridden and redirected to the logger.
    """

    configure_default_serve_logger()
    serve_logger = logging.getLogger("ray.serve")
    assert len(serve_logger.handlers) == 1
    assert isinstance(serve_logger.handlers[0], logging.StreamHandler)
    assert print != redirected_print
    assert not isinstance(sys.stdout, StreamToLogger)
    assert not isinstance(sys.stderr, StreamToLogger)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))

load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//bazel:python.bzl", "py_test_module_list")

py_library(
    name = "conftest",
    srcs = ["conftest.py"],
)

py_library(
    name = "common",
    srcs = glob(["common/*.py"]),
    visibility = [
        "//python/ray/serve/tests:__subpackages__",
    ],
)

# Minimal installation test (should *not* include conftest).
py_test_module_list(
    size = "small",
    files = [
        "test_minimal_installation.py",
    ],
    tags = [
        "exclusive",
        "minimal",
        "team:serve",
    ],
    deps = [
        "//python/ray/serve:serve_lib",
    ],
)

# Small tests.
py_test_module_list(
    size = "small",
    files = [
        "test_advanced.py",
        "test_cluster_node_info_cache.py",
        "test_constructor_failure.py",
        "test_controller.py",
        "test_deployment_version.py",
        "test_enable_task_events.py",
        "test_expected_versions.py",
        "test_http_cancellation.py",
        "test_kv_store.py",
        "test_long_poll.py",
        "test_persistence.py",
        "test_proxy_actor_wrapper.py",
        "test_replica_request_context.py",
        "test_util.py",
        "test_websockets.py",
    ],
    tags = [
        "exclusive",
        "team:serve",
    ],
    deps = [
        ":common",
        ":conftest",
        "//python/ray/serve:serve_lib",
    ],
)

# Medium tests.
py_test_module_list(
    size = "medium",
    files = [
        "test_actor_replica_wrapper.py",
        "test_backpressure.py",
        "test_batching.py",
        "test_callback.py",
        "test_cluster.py",
        "test_controller_recovery.py",
        "test_deploy_2.py",
        "test_deployment_scheduler.py",
        "test_failure.py",
        "test_handle_1.py",
        "test_handle_2.py",
        "test_handle_cancellation.py",
        "test_handle_streaming.py",
        "test_healthcheck.py",
        "test_http_headers.py",
        "test_http_routes.py",
        "test_max_replicas_per_node.py",
        "test_multiplex.py",
        "test_proxy.py",
        "test_proxy_response_generator.py",
        "test_ray_client.py",
        "test_record_routing_stats.py",
        "test_regression.py",
        "test_replica_placement_group.py",
        "test_request_timeout.py",
        "test_streaming_response.py",
        "test_target_capacity.py",
        "test_telemetry.py",
    ],
    tags = [
        "exclusive",
        "team:serve",
    ],
    deps = [
        ":common",
        ":conftest",
        "//python/ray/serve:serve_lib",
    ],
)

# Medium tests, don't run on windows.
py_test_module_list(
    size = "medium",
    files = [
        "test_fastapi.py",
        "test_gcs_failure.py",
        "test_gradio.py",
    ],
    tags = [
        "exclusive",
        "no_windows",
        "team:serve",
    ],
    deps = [
        ":common",
        ":conftest",
        "//python/ray/serve:serve_lib",
    ],
)

# Large tests.
py_test_module_list(
    size = "large",
    files = [
        "test_autoscaling_policy.py",
        "test_deploy.py",
        "test_grpc.py",
        "test_logging.py",
        "test_standalone.py",
        "test_standalone_3.py",
        "test_telemetry_1.py",
        "test_telemetry_2.py",
    ],
    tags = [
        "exclusive",
        "team:serve",
    ],
    deps = [
        ":common",
        ":conftest",
        "//python/ray/serve:serve_lib",
    ],
)

# Large tests requiring `test_config_files/`.
py_test_module_list(
    size = "large",
    data = glob(["test_config_files/**/*"]),
    files = [
        "test_cli.py",
        "test_cli_2.py",
        "test_cli_3.py",
    ],
    tags = [
        "exclusive",
        "team:serve",
    ],
    deps = [
        ":common",
        ":conftest",
        "//python/ray/serve:serve_lib",
    ],
)

# Large tests require `test_config_files/`, no windows.
py_test_module_list(
    size = "large",
    data = glob(["test_config_files/**/*"]),
    files = [
        "test_standalone_2.py",
    ],
    tags = [
        "exclusive",
        "no_windows",
        "team:serve",
    ],
    deps = [
        ":common",
        ":conftest",
        "//python/ray/serve:serve_lib",
    ],
)

# Run serially on Windows.
py_test_module_list(
    size = "medium",
    timeout = "long",
    files = [
        "test_deploy_app.py",
        "test_deploy_app_2.py",
        "test_metrics.py",
        "test_metrics_2.py",
    ],
    tags = [
        "exclusive",
        "team:serve",
        "use_all_core_windows",
    ],
    deps = [
        ":common",
        ":conftest",
        "//python/ray/serve:serve_lib",
    ],
)

# Minimal tests
py_test_module_list(
    size = "large",
    files = [
        "test_api.py",
        "test_api_2.py",
        "test_model_composition.py",
    ],
    tags = [
        "exclusive",
        "minimal",
        "team:serve",
    ],
    deps = [
        ":common",
        ":conftest",
        "//python/ray/serve:serve_lib",
    ],
)

# Post-wheel-build tests.
py_test_module_list(
    size = "large",
    files = [
        "test_runtime_env.py",
        "test_runtime_env_2.py",
    ],
    tags = [
        "exclusive",
        "post_wheel_build",
        "team:serve",
    ],
    deps = [
        ":common",
        ":conftest",
        "//python/ray/serve:serve_lib",
    ],
)

# Runs test_api and test_failure with injected failures in the controller.
py_test(
    name = "test_controller_crashes",
    size = "large",
    srcs = [
        "test_api.py",
        "test_api_2.py",
        "test_controller_crashes.py",
        "test_failure.py",
    ],
    tags = [
        "exclusive",
        "team:serve",
    ],
    deps = [
        ":common",
        ":conftest",
        "//python/ray/serve:serve_lib",
    ],
)

# Serve HA.
py_test(
    name = "test_serve_ha",
    size = "medium",
    srcs = ["test_serve_ha.py"],
    tags = [
        "exclusive",
        "ha_integration",
        "team:serve",
    ],
    deps = [
        ":common",
        ":conftest",
        "//python/ray/serve:serve_lib",
    ],
)

# ----- TEST FEATURE FLAGS -----

# Test autoscaling with metrics collected from replica instead of handle.
py_test(
    name = "test_autoscaling_policy_with_metr_disab",
    size = "large",
    srcs = ["test_autoscaling_policy.py"],
    env = {
        "RAY_SERVE_COLLECT_AUTOSCALING_METRICS_ON_HANDLE": "0",
        # Make sure queued metrics are cleared out quickly.
        "RAY_SERVE_HANDLE_METRIC_PUSH_INTERVAL_S": "0.1",
    },
    main = "test_autoscaling_policy.py",
    tags = [
        "autoscaling",
        "exclusive",
        "team:serve",
    ],
    deps = [
        ":common",
        ":conftest",
        "//python/ray/serve:serve_lib",
    ],
)

# Test feature flag for task events.
py_test_module_list(
    size = "small",
    data = glob(["test_config_files/**/*"]),
    env = {"RAY_SERVE_ENABLE_TASK_EVENTS": "1"},
    files = [
        "test_enable_task_events.py",
    ],
    name_suffix = "_with_task_events_enabled",
    tags = [
        "exclusive",
        "no_windows",
        "team:serve",
    ],
    deps = [
        ":common",
        ":conftest",
        "//python/ray/serve:serve_lib",
    ],
)

# Medium tests with compact scheduling
py_test_module_list(
    size = "medium",
    data = glob(["test_config_files/**/*"]),
    env = {"RAY_SERVE_USE_COMPACT_SCHEDULING_STRATEGY": "1"},
    files = [
        "test_cluster.py",
        "test_controller_recovery.py",
        "test_deployment_scheduler.py",
        "test_gcs_failure.py",
        "test_max_replicas_per_node.py",
        "test_replica_placement_group.py",
    ],
    name_suffix = "_with_compact_scheduling",
    tags = [
        "exclusive",
        "no_windows",
        "team:serve",
    ],
    deps = [
        ":common",
        ":conftest",
        "//python/ray/serve:serve_lib",
    ],
)

# Large tests with compact scheduling
py_test_module_list(
    size = "large",
    env = {"RAY_SERVE_USE_COMPACT_SCHEDULING_STRATEGY": "1"},
    files = [
        "test_standalone.py",
        "test_standalone_3.py",
    ],
    name_suffix = "_with_comp_sche",
    tags = [
        "exclusive",
        "team:serve",
    ],
    deps = [
        ":common",
        ":conftest",
        "//python/ray/serve:serve_lib",
    ],
)

# Large tests with compact scheduling, no windows
py_test_module_list(
    size = "large",
    data = glob(["test_config_files/**/*"]),
    env = {"RAY_SERVE_USE_COMPACT_SCHEDULING_STRATEGY": "1"},
    files = [
        "test_standalone_2.py",
    ],
    name_suffix = "_with_compact_scheduling",
    tags = [
        "exclusive",
        "no_windows",
        "team:serve",
    ],
    deps = [
        ":common",
        ":conftest",
        "//python/ray/serve:serve_lib",
    ],
)

# Test handle API with local testing mode.
py_test_module_list(
    size = "small",
    env = {"RAY_SERVE_FORCE_LOCAL_TESTING_MODE": "1"},
    files = [
        "test_handle_1.py",
        "test_handle_2.py",
        "test_handle_cancellation.py",
        "test_handle_streaming.py",
    ],
    name_suffix = "_with_local_testing_mode",
    tags = [
        "exclusive",
        "no_windows",
        "team:serve",
    ],
    deps = [
        ":common",
        ":conftest",
        "//python/ray/serve:serve_lib",
    ],
)

# Test currently off-by-default behavior to run replica sync methods in a threadpool.
# TODO(edoakes): remove this once the FF is flipped on by default.
py_test_module_list(
    size = "small",
    env = {"RAY_SERVE_RUN_SYNC_IN_THREADPOOL": "1"},
    files = [
        "test_replica_sync_methods.py",
    ],
    name_suffix = "_with_run_sync_in_threadpool",
    tags = [
        "exclusive",
        "no_windows",
        "team:serve",
    ],
    deps = [
        ":common",
        ":conftest",
        "//python/ray/serve:serve_lib",
    ],
)

py_test_module_list(
    size = "medium",
    env = {"RAY_SERVE_RUN_ROUTER_IN_SEPARATE_LOOP": "0"},
    files = [
        "test_handle_same_loop.py",
        "test_proxy.py",
    ],
    name_suffix = "_with_router_in_same_loop",
    tags = [
        "exclusive",
        "no_windows",
        "team:serve",
    ],
    deps = [
        ":common",
        ":conftest",
        "//python/ray/serve:serve_lib",
    ],
)

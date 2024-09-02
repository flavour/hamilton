from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid
from types import ModuleType
from typing import Any

import hamilton.lifecycle.base as lifecycle_base
from hamilton import base, driver, graph, lifecycle, node, telemetry
from hamilton.async_utils import await_dict_of_tasks, process_value
from hamilton.execution import executors, graph_functions, grouping, state
from hamilton.io.materialization import ExtractorFactory, MaterializerFactory

logger = logging.getLogger(__name__)


class AsyncGraphAdapter(lifecycle_base.BaseDoNodeExecute, lifecycle.ResultBuilder):
    """Graph adapter for use with the :class:`AsyncDriver` class."""

    def __init__(
        self,
        result_builder: base.ResultMixin = None,
        async_lifecycle_adapters: lifecycle_base.LifecycleAdapterSet | None = None,
    ):
        """Creates an AsyncGraphAdapter class. Note this will *only* work with the AsyncDriver class.

        Some things to note:

            1. This executes everything at the end (recursively). E.G. the final DAG nodes are awaited
            2. This does *not* work with decorators when the async function is being decorated. That is\
            because that function is called directly within the decorator, so we cannot await it.
        """
        super(AsyncGraphAdapter, self).__init__()
        self.adapter = (
            async_lifecycle_adapters
            if async_lifecycle_adapters is not None
            else lifecycle_base.LifecycleAdapterSet()
        )
        self.result_builder = result_builder if result_builder else base.PandasDataFrameResult()
        self.is_initialized = False

    def do_node_execute(
        self,
        *,
        run_id: str,
        node_: node.Node,
        kwargs: dict[str, Any],
        task_id: str | None = None,
    ) -> Any:
        """Executes a node. Note this doesn't actually execute it -- rather, it returns a task.
        This does *not* use async def, as we want it to be awaited on later -- this await is done
        in processing parameters of downstream functions/final results. We can ensure that as
        we also run the driver that this corresponds to.

        Note that this assumes that everything is awaitable, even if it isn't.
        In that case, it just wraps it in one.

        :param task_id:
        :param node_:
        :param run_id:
        :param node: Node to wrap
        :param kwargs: Keyword arguments (either coroutines or raw values) to call it with
        :return: A task
        """
        callabl = node_.callable

        async def new_fn(fn=callabl, **fn_kwargs):
            task_dict = {key: process_value(value) for key, value in fn_kwargs.items()}
            fn_kwargs = await await_dict_of_tasks(task_dict)
            error = None
            result = None
            success = True
            pre_node_execute_errored = False
            try:
                if self.adapter.does_hook("pre_node_execute", is_async=True):
                    try:
                        await self.adapter.call_all_lifecycle_hooks_async(
                            "pre_node_execute",
                            run_id=run_id,
                            node_=node_,
                            kwargs=fn_kwargs,
                            task_id=task_id,
                        )
                    except Exception as e:
                        pre_node_execute_errored = True
                        raise e
                # TODO -- consider how to use node execution methods in the future
                # This is messy as it is a method called within a method...
                # if self.adapter.does_method("do_node_execute", is_async=False):
                #     result = self.adapter.call_lifecycle_method_sync(
                #         "do_node_execute",
                #         run_id=run_id,
                #         node_=node_,
                #         kwargs=kwargs,
                #         task_id=task_id,
                #     )
                # else:

                result = (
                    await fn(**fn_kwargs) if asyncio.iscoroutinefunction(fn) else fn(**fn_kwargs)
                )
            except Exception as e:
                success = False
                error = e
                step = "[pre-node-execute:async]" if pre_node_execute_errored else ""
                message = graph_functions.create_error_message(kwargs, node_, step)
                logger.exception(message)
                raise
            finally:
                if not pre_node_execute_errored and self.adapter.does_hook(
                    "post_node_execute", is_async=True
                ):
                    try:
                        await self.adapter.call_all_lifecycle_hooks_async(
                            "post_node_execute",
                            run_id=run_id,
                            node_=node_,
                            kwargs=fn_kwargs,
                            success=success,
                            error=error,
                            result=result,
                            task_id=task_id,
                        )
                    except Exception:
                        message = graph_functions.create_error_message(
                            kwargs, node_, "[post-node-execute]"
                        )
                        logger.exception(message)
                        raise

            return result

        coroutine = new_fn(**kwargs)
        task = asyncio.create_task(coroutine)
        return task

    def build_result(self, **outputs: Any) -> Any:
        return self.result_builder.build_result(**outputs)


def separate_sync_from_async(
    adapters: list[lifecycle.LifecycleAdapter],
) -> tuple[list[lifecycle.LifecycleAdapter], list[lifecycle.LifecycleAdapter]]:
    """Separates the sync and async adapters from a list of adapters.
    Note this only works with hooks -- we'll be dealing with methods later.

    :param adapters: List of adapters
    :return: Tuple of sync adapters, async adapters
    """

    adapter_set = lifecycle_base.LifecycleAdapterSet(*adapters)
    # this is using internal(ish) fields (.sync_hooks/.async_hooks) -- we should probably expose it
    # For now this is OK
    # Note those are dict[hook_name, list[hook]], so we have to flatten
    return (
        [sync_adapter for adapters in adapter_set.sync_hooks.values() for sync_adapter in adapters],
        [
            async_adapter
            for adapters in adapter_set.async_hooks.values()
            for async_adapter in adapters
        ],
    )


class AsyncTaskBasedGraphExecutor(driver.GraphExecutor):
    def validate(self, nodes_to_execute: list[node.Node]):
        """Currently this can run every valid graph"""
        pass

    def __init__(
        self,
        execution_manager: executors.ExecutionManager,
        grouping_strategy: grouping.GroupingStrategy,
        adapter: lifecycle_base.LifecycleAdapterSet,
    ):
        """Executor for task-based execution. This enables grouping of nodes into tasks, as
        well as parallel execution/dynamic spawning of nodes.

        :param execution_manager: Utility to assign task executors to node groups
        :param grouping_strategy: Utility to group nodes into tasks
        :param result_builder: Utility to build the final result"""

        self.execution_manager = execution_manager
        self.grouping_strategy = grouping_strategy
        self.adapter = adapter

    async def execute(
        self,
        fg: graph.FunctionGraph,
        final_vars: list[str],
        overrides: dict[str, Any],
        inputs: dict[str, Any],
        run_id: str,
    ) -> dict[str, Any]:
        """Executes a graph, task by task. This blocks until completion.

        This does the following:
        1. Groups the nodes into tasks
        2. Creates an execution state and a results cache
        3. Runs it to completion, populating the results cache
        4. Returning the results from the results cache
        """
        inputs = graph_functions.combine_config_and_inputs(fg.config, inputs)
        (
            transform_nodes_required_for_execution,
            user_defined_nodes_required_for_execution,
        ) = fg.get_upstream_nodes(final_vars, runtime_inputs=inputs, runtime_overrides=overrides)

        all_nodes_required_for_execution = list(
            set(transform_nodes_required_for_execution).union(
                user_defined_nodes_required_for_execution
            )
        )
        grouped_nodes = self.grouping_strategy.group_nodes(
            all_nodes_required_for_execution
        )  # pure function transform
        # Instantiate a result cache so we can use later
        # Pass in inputs so we can pre-populate the results cache
        prehydrated_results = {**overrides, **inputs}
        results_cache = state.DictBasedResultCache(prehydrated_results)
        # Create tasks from the grouped nodes, filtering/pruning as we go
        tasks = grouping.create_task_plan(grouped_nodes, final_vars, overrides, self.adapter)
        # Create a task graph and execution state
        execution_state = state.ExecutionState(
            tasks, results_cache, run_id
        )  # Stateful storage for the DAG
        # Blocking call to run through until completion
        await executors.run_graph_to_completion_async(execution_state, self.execution_manager)
        # Read the final variables from the result cache
        raw_result = results_cache.read(final_vars)
        return raw_result


class AsyncDriver(driver.Driver):
    """Async driver. This is a driver that uses the AsyncGraphAdapter to execute the graph.

    .. code-block:: python

        dr = async_driver.AsyncDriver({}, async_module, result_builder=base.DictResult())
        df = await dr.execute([...], inputs=...)

    """

    def __init__(
        self,
        config,
        *modules,
        result_builder: base.ResultMixin | None = None,
        adapters: list[lifecycle.LifecycleAdapter] | None = None,
        _graph_executor: driver.GraphExecutor | None = None,
    ):
        """Instantiates an asynchronous driver.

        You will also need to call `ainit` to initialize the driver if you have any hooks/adapters.

        Note that this is not the desired API -- you should be using the :py:class:`hamilton.async_driver.Builder` class to create the driver.

        This will only (currently) work properly with asynchronous lifecycle hooks, and does not support methods or validators.
        You can still pass in synchronous lifecycle hooks, but they may behave strangely.

        :param config: Config to build the graph
        :param modules: Modules to crawl for fns/graph nodes
        :param adapters: Adapters to use for lifecycle methods.
        :param result_builder: Results mixin to compile the graph's final results. TBD whether this should be included in the long run.
        :param _graph_executor: Not public facing, do not use this parameter. This is injected by the builder.
            If you need to tune execution, use the builder to do so.
        """
        if adapters is None:
            adapters = []
        sync_adapters, async_adapters = separate_sync_from_async(adapters)

        if _graph_executor is None:
            # we'll need to use this in multiple contexts so we'll keep it around for later
            result_builders = [
                adapter for adapter in adapters if isinstance(adapter, base.ResultMixin)
            ]
            if result_builder is not None:
                result_builders.append(result_builder)
            if len(result_builders) > 1:
                raise ValueError(
                    "You cannot pass more than one result builder to the async driver. "
                    "Please pass in a single result builder"
                )
            # it will be defaulted by the graph adapter
            result_builder = result_builders[0] if len(result_builders) == 1 else None

            # We pass in the async adapters here as this can call node-level hooks
            # Otherwise we trust the driver/fn graph to call sync adapters
            _graph_executor = AsyncGraphAdapter(
                result_builder=result_builder,
                async_lifecycle_adapters=lifecycle_base.LifecycleAdapterSet(*async_adapters),
            )

        super(AsyncDriver, self).__init__(
            config,
            *modules,
            adapter=[
                _graph_executor,
                # We pass in the sync adapters here as this can call
                *sync_adapters,
                *async_adapters,  # note async adapters will not be called during synchronous execution -- this is for access later
            ],
        )
        self.initialized = False

    async def ainit(self) -> "AsyncDriver":
        """Initializes the driver when using async. This only exists for backwards compatibility.
        In Hamilton 2.0, we will be using an asynchronous constructor.
        See https://dev.to/akarshan/asynchronous-python-magic-how-to-create-awaitable-constructors-with-asyncmixin-18j5.
        """
        if self.initialized:
            # this way it can be called twice
            return self
        if self.adapter.does_hook("post_graph_construct", is_async=True):
            await self.adapter.call_all_lifecycle_hooks_async(
                "post_graph_construct",
                graph=self.graph,
                modules=self.graph_modules,
                config=self.config,
            )
        await self.adapter.ainit()
        self.initialized = True
        return self

    async def raw_execute(
        self,
        final_vars: list[str],
        overrides: dict[str, Any] | None = None,
        display_graph: bool = False,  # don't care
        inputs: dict[str, Any] | None = None,
        _fn_graph: graph.FunctionGraph | None = None,
    ) -> dict[str, Any]:
        """Executes the graph, returning a dictionary of strings (node keys) to final results.

        :param final_vars: Variables to execute (+ upstream)
        :param overrides: Overrides for nodes
        :param display_graph: whether or not to display graph -- this is not supported.
        :param inputs:  Inputs for DAG runtime calculation
        :param _fn_graph: Function graph for compatibility with superclass -- unused
        :return: A dict of key -> result
        """
        assert _fn_graph is None, (
            "_fn_graph must not be provided "
            "-- the only reason you'd do this is to use materialize(), which is not supported yet.."
        )
        run_id = str(uuid.uuid4())
        nodes, user_nodes = self.graph.get_upstream_nodes(final_vars, inputs, overrides)
        memoized_computation = dict()  # memoized storage
        if self.adapter.does_hook("pre_graph_execute"):
            await self.adapter.call_all_lifecycle_hooks_sync_and_async(
                "pre_graph_execute",
                run_id=run_id,
                graph=self.graph,
                final_vars=final_vars,
                inputs=inputs,
                overrides=overrides,
            )
        results = None
        error = None
        success = False
        try:
            self.graph.execute(nodes, memoized_computation, overrides, inputs, run_id=run_id)
            if display_graph:
                raise ValueError(
                    "display_graph=True is not supported for the async graph adapter. "
                    "Instead you should be using visualize_execution."
                )
            task_dict = {
                key: asyncio.create_task(process_value(memoized_computation[key]))
                for key in final_vars
            }
            results = await await_dict_of_tasks(task_dict)
            success = True
        except Exception as e:
            error = e
            success = False
            raise e
        finally:
            if self.adapter.does_hook("post_graph_execute", is_async=None):
                await self.adapter.call_all_lifecycle_hooks_sync_and_async(
                    "post_graph_execute",
                    run_id=run_id,
                    graph=self.graph,
                    success=success,
                    error=error,
                    results=results,
                )
        return results

    async def execute(
        self,
        final_vars: list[str],
        overrides: dict[str, Any] | None = None,
        display_graph: bool = False,
        inputs: dict[str, Any] | None = None,
    ) -> Any:
        """Executes computation.

        :param final_vars: the final list of variables we want to compute.
        :param overrides: values that will override "nodes" in the DAG.
        :param display_graph: DEPRECATED. Whether we want to display the graph being computed.
        :param inputs: Runtime inputs to the DAG.
        :return: an object consisting of the variables requested, matching the type returned by the GraphAdapter.
            See constructor for how the GraphAdapter is initialized. The default one right now returns a pandas
            dataframe.
        """
        if display_graph:
            raise ValueError(
                "display_graph=True is not supported for the async graph adapter. "
                "Instead you should be using visualize_execution."
            )
        start_time = time.time()
        run_successful = True
        error = None
        _final_vars = self._create_final_vars(final_vars)
        try:
            outputs = await self.raw_execute(_final_vars, overrides, display_graph, inputs=inputs)
            # Currently we don't allow async build results, but we could.
            if self.adapter.does_method("do_build_result", is_async=False):
                return self.adapter.call_lifecycle_method_sync("do_build_result", outputs=outputs)
            return outputs
        except Exception as e:
            run_successful = False
            logger.error(driver.SLACK_ERROR_MESSAGE)
            error = telemetry.sanitize_error(*sys.exc_info())
            raise e
        finally:
            duration = time.time() - start_time
            # ensure we can capture telemetry in async friendly way.
            if telemetry.is_telemetry_enabled():

                async def make_coroutine():
                    self.capture_execute_telemetry(
                        error, final_vars, inputs, overrides, run_successful, duration
                    )

                try:
                    # we don't have to await because we are running within the event loop.
                    asyncio.create_task(make_coroutine())
                except Exception as e:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.error(f"Encountered error submitting async telemetry:\n{e}")

    def capture_constructor_telemetry(
        self,
        error: str | None,
        modules: tuple[ModuleType],
        config: dict[str, Any],
        adapter: base.HamiltonGraphAdapter,
    ):
        """Ensures we capture constructor telemetry the right way in an async context.

        This is a simpler wrapper around what's in the driver class.

        :param error: sanitized error string, if any.
        :param modules: tuple of modules to build DAG from.
        :param config: config to create the driver.
        :param adapter: adapter class object.
        """
        if telemetry.is_telemetry_enabled():
            try:
                # check whether the event loop has been started yet or not
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.run_in_executor(
                        None,
                        super(AsyncDriver, self).capture_constructor_telemetry,
                        error,
                        modules,
                        config,
                        adapter,
                    )
                else:

                    async def make_coroutine():
                        super(AsyncDriver, self).capture_constructor_telemetry(
                            error, modules, config, adapter
                        )

                    loop.run_until_complete(make_coroutine())
            except Exception as e:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.error(f"Encountered error submitting async telemetry:\n{e}")


class Builder(driver.Builder):
    """Builder for the async driver. This is equivalent to the standard builder, but has a more limited API.
    Note this does not support dynamic execution or materializers (for now).

    Here is an example of how you might use it to get the tracker working:

    .. code-block:: python

        from hamilton_sdk import tracker

        tracker_async = adapters.AsyncHamiltonTracker(
            project_id=1,
            username="elijah",
            dag_name="async_tracker",
        )
        dr = (
            await async_driver.Builder()
            .with_modules(async_module)
            .with_adapters(tracking_async)
            .build()
        )
    """

    def __init__(self):
        super(Builder, self).__init__()

    def _not_supported(self, method_name: str, additional_message: str = ""):
        raise ValueError(
            f"Builder().{method_name}() is not supported for the async driver. {additional_message}"
        )

    def enable_dynamic_execution(self, *, allow_experimental_mode: bool = False) -> "Builder":
        """Enables the Parallelizable[] type, which in turn enables:
        1. Grouped execution into tasks
        2. Parallel execution
        :return: self
        """
        if not allow_experimental_mode:
            raise ValueError(
                "Remote execution is currently experimental. "
                "Please set allow_experimental_mode=True to enable it."
            )
        self.v2_executor = True
        return self

    def with_materializers(
        self, *materializers: ExtractorFactory | MaterializerFactory
    ) -> "Builder":
        self._not_supported("with_materializers")

    def with_adapter(self, adapter: base.HamiltonGraphAdapter) -> "Builder":
        self._not_supported(
            "with_adapter",
            "Use with_adapters instead to pass in the tracker (or other async hooks/methods)",
        )

    def build_without_init(self) -> AsyncDriver:
        """Allows you to build the async driver without initialization. Use this at
        your own risk -- we highly recommend calling `.ainit` on the final result.

        :return:
        """
        adapters = self.adapters if self.adapters is not None else []
        if self.legacy_graph_adapter is not None:
            adapters.append(self.legacy_graph_adapter)

        if self.v2_executor:
            execution_manager = self.execution_manager
            if execution_manager is None:
                local_executor = self.local_executor or executors.AsynchronousLocalTaskExecutor()
                remote_executor = self.remote_executor or executors.MultiThreadingExecutor(
                    max_tasks=10
                )
                execution_manager = executors.DefaultExecutionManager(
                    local_executor=local_executor, remote_executor=remote_executor
                )
            grouping_strategy = self.grouping_strategy or grouping.GroupByRepeatableBlocks()
            sync_adapters, async_adapters = separate_sync_from_async(adapters)
            graph_executor = AsyncTaskBasedGraphExecutor(
                execution_manager=execution_manager,
                grouping_strategy=grouping_strategy,
                adapter=lifecycle_base.LifecycleAdapterSet(*async_adapters),
            )

            return AsyncDriver(
                self.config,
                *self.modules,
                adapters=adapters,
                # _materializers=self.materializers,
                _graph_executor=graph_executor,
                # _use_legacy_adapter=False,
            )
        else:
            # We should really be doing this in the constructor
            # but the AsyncGraphAdapter originally used the pandas builder
            # so we pass in the right one to ensure backwards compatibility
            # This will become the default API soon, so it's OK to put the complexity here
            result_builders = [
                adapter for adapter in adapters if isinstance(adapter, base.ResultMixin)
            ]
            specified_result_builder = base.DictResult() if len(result_builders) == 0 else None

            return AsyncDriver(
                self.config,
                *self.modules,
                adapters=self.adapters,
                result_builder=specified_result_builder,
            )

    async def build(self):
        """Builds the async driver. This also initializes it, hence the async definition.
        If you don't want to use async, you can use `build_without_init` and call `ainit` later,
        but we recommend using this in an asynchronous lifespan management function (E.G. in fastAPI),
        or something similar.

        :return: The fully
        """
        dr = self.build_without_init()
        return await dr.ainit()

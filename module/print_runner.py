from langchain_core.tracers.schemas import Run

def printStart(runner: Run):
    print("start", runner.metadata, runner.inputs)


def printEnd(runner: Run):
    print("end", runner.metadata, runner.inputs)
    if runner.inputs["context"] is not None:
        print("retrieved: ", len(runner.inputs["context"]), " documents")

def with_print_listener(target):
    return target.with_listeners(on_start=printStart, on_end=printEnd)
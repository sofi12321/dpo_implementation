from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def load_llama(n_gpu_layers = 40, n_batch = 512, model_path="/content/llama-2-7b-chat.Q4_K_M.gguf", temperature=0.1, verbose=True):

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm_evaluator = LlamaCpp(
        model_path=model_path,
        temperature=temperature,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=verbose,
    )
    return llm_evaluator


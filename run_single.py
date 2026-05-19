from run_benchmark import run_strategy, _make_memory
from llm.llm_groq import SimpleGroqClient
from memory_strategies import BaselineMemory
from llm.utils import load_scripts

# Use a fresh key (Benchmark2)
llm = SimpleGroqClient(["GROQ_API_KEY_2"])
script = load_scripts("test_scripts")[0]  # legal_01
results, history = run_strategy(
    script=script,
    strategy_name="baseline",
    strategy_class=BaselineMemory,
    strategy_kwargs={},
    llm=llm,
    script_id="legal_01",
    rep=4,   # only repetition 4
    memory=None,
)
# Append to existing CSV manually or just save to a new CSV
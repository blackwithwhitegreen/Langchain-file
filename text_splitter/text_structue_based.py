from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
We introduce MiniMax-M1, the world's first open-weight, large-scale hybrid-attention reasoning model. MiniMax-M1 is powered by a hybrid Mixture-of-Experts (MoE) architecture combined with a lightning attention mechanism. The model is developed based on our previous MiniMax-Text-01 model, which contains a total of 456 billion parameters with 45.9 billion parameters activated per token. Consistent with MiniMax-Text-01, the M1 model natively supports a context length of 1 million tokens, 8x the context size of DeepSeek R1. Furthermore, the lightning attention mechanism in MiniMax-M1 enables efficient scaling of test-time compute – For example, compared to DeepSeek R1, M1 consumes 25% of the FLOPs at a generation length of 100K tokens. These properties make M1 particularly suitable for complex tasks that require processing long inputs and thinking extensively. MiniMax-M1 is trained using large-scale reinforcement learning (RL) on diverse problems ranging from traditional mathematical reasoning to sandbox-based, real-world software engineering environments. We develop an efficient RL scaling framework for M1 highlighting two perspectives: (1) We propose CISPO, a novel algorithm that clips importance sampling weights instead of token updates, which outperforms other competitive RL variants; (2) Our hybrid-attention design naturally enhances the efficiency of RL, where we address unique challenges when scaling RL with the hybrid architecture. We train two versions of MiniMax-M1 models with 40K and 80K thinking budgets respectively. Experiments on standard benchmarks show that our models outperform other strong open-weight models such as the original DeepSeek-R1 and Qwen3-235B, particularly on complex software engineering, tool using, and long context tasks. With efficient scaling of test-time compute, MiniMax-M1 serves as a strong foundation for next-generation language model agents to reason and tackle real-world challenges. 
"""

# Initalize the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0

)

# Perform the split
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_YourRealTokenHere"


load_dotenv(find_dotenv())

# client = InferenceClient(
#     provider="hf-inference"
    
# )


model_1 = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=0.5
)
# model_2 = ChatHuggingFace.client(
#     model="microsoft/Phi-3-mini-4k-instruct",
#     task = 'text-generation'
# )

model_2 = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.6,
    max_tokens=64,
    # api_key=groq_api_key
)

prompt1 = PromptTemplate(
    template="Generate short and simple nots from the following text \n{text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notex ->{notes} and quiz -> {quiz}",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parllel_chain = RunnableParallel({
    'notes': prompt1 | model_1 | parser,
    'quiz': prompt2 | model_2 | parser
})

merge_chain = prompt3 | model_1 | parser

chain = parllel_chain | merge_chain

text = """
Long short-term memory (LSTM)[1] is a type of recurrent neural network (RNN) aimed at mitigating the vanishing gradient problem[2] commonly encountered by traditional RNNs. Its relative insensitivity to gap length is its advantage over other RNNs, hidden Markov models, and other sequence learning methods. It aims to provide a short-term memory for RNN that can last thousands of timesteps (thus "long short-term memory").[1] The name is made in analogy with long-term memory and short-term memory and their relationship, studied by cognitive psychologists since the early 20th century.

An LSTM unit is typically composed of a cell and three gates: an input gate, an output gate,[3] and a forget gate.[4] The cell remembers values over arbitrary time intervals, and the gates regulate the flow of information into and out of the cell. Forget gates decide what information to discard from the previous state, by mapping the previous state and the current input to a value between 0 and 1. A (rounded) value of 1 signifies retention of the information, and a value of 0 represents discarding. Input gates decide which pieces of new information to store in the current cell state, using the same system as forget gates. Output gates control which pieces of information in the current cell state to output, by assigning a value from 0 to 1 to the information, considering the previous and current states. Selectively outputting relevant information from the current state allows the LSTM network to maintain useful, long-term dependencies to make predictions, both in current and future time-steps.

LSTM has wide applications in classification,[5][6] data processing, time series analysis tasks,[7] speech recognition,[8][9] machine translation,[10][11] speech activity detection,[12] robot control,[13][14] video games,[15][16] healthcare.[17]
Motivation

In theory, classic RNNs can keep track of arbitrary long-term dependencies in the input sequences. The problem with classic RNNs is computational (or practical) in nature: when training a classic RNN using back-propagation, the long-term gradients which are back-propagated can "vanish", meaning they can tend to zero due to very small numbers creeping into the computations, causing the model to effectively stop learning. RNNs using LSTM units partially solve the vanishing gradient problem, because LSTM units allow gradients to also flow with little to no attenuation. However, LSTM networks can still suffer from the exploding gradient problem.[18]

The intuition behind the LSTM architecture is to create an additional module in a neural network that learns when to remember and when to forget pertinent information.[4] In other words, the network effectively learns which information might be needed later on in a sequence and when that information is no longer needed. For instance, in the context of natural language processing, the network can learn grammatical dependencies.[19] An LSTM might process the sentence "Dave, as a result of his controversial claims, is now a pariah" by remembering the (statistically likely) grammatical gender and number of the subject Dave, note that this information is pertinent for the pronoun his and note that this information is no longer important after the verb is.
Variants

In the equations below, the lowercase variables represent vectors. Matrices W q {\displaystyle W_{q}} and U q {\displaystyle U_{q}} contain, respectively, the weights of the input and recurrent connections, where the subscript q {\displaystyle _{q}} can either be the input gate i {\displaystyle i}, output gate o {\displaystyle o}, the forget gate f {\displaystyle f} or the memory cell c {\displaystyle c}, depending on the activation being calculated. In this section, we are thus using a "vector notation". So, for example, c t ∈ R h {\displaystyle c_{t}\in \mathbb {R} ^{h}} is not just one unit of one LSTM cell, but contains h {\displaystyle h} LSTM cell's units.

See [20] for an empirical study of 8 architectural variants of LSTM.
LSTM with a forget gate


Each of the gates can be thought as a "standard" neuron in a feed-forward (or multi-layer) neural network: that is, they compute an activation (using an activation function) of a weighted sum. i t , o t {\displaystyle i_{t},o_{t}} and f t {\displaystyle f_{t}} represent the activations of respectively the input, output and forget gates, at time step t {\displaystyle t}.

The 3 exit arrows from the memory cell c {\displaystyle c} to the 3 gates i , o {\displaystyle i,o} and f {\displaystyle f} represent the peephole connections. These peephole connections actually denote the contributions of the activation of the memory cell c {\displaystyle c} at time step t − 1 {\displaystyle t-1}, i.e. the contribution of c t − 1 {\displaystyle c_{t-1}} (and not c t {\displaystyle c_{t}}, as the picture may suggest). In other words, the gates i , o {\displaystyle i,o} and f {\displaystyle f} calculate their activations at time step t {\displaystyle t} (i.e., respectively, i t , o t {\displaystyle i_{t},o_{t}} and f t {\displaystyle f_{t}}) also considering the activation of the memory cell c {\displaystyle c} at time step t − 1 {\displaystyle t-1}, i.e. c t − 1 {\displaystyle c_{t-1}}.

The single left-to-right arrow exiting the memory cell is not a peephole connection and denotes c t {\displaystyle c_{t}}.

An RNN using LSTM units can be trained in a supervised fashion on a set of training sequences, using an optimization algorithm like gradient descent combined with backpropagation through time to compute the gradients needed during the optimization process, in order to change each weight of the LSTM network in proportion to the derivative of the error (at the output layer of the LSTM network) with respect to corresponding weight.

A problem with using gradient descent for standard RNNs is that error gradients vanish exponentially quickly with the size of the time lag between important events. This is due to lim n → ∞ W n = 0 {\displaystyle \lim _{n\to \infty }W^{n}=0} if the spectral radius of W {\displaystyle W} is smaller than 1.[2][24]

However, with LSTM units, when error values are back-propagated from the output layer, the error remains in the LSTM unit's cell. This "error carousel" continuously feeds error back to each of the LSTM unit's gates, until they learn to cut off the value.
CTC score function

Many applications use stacks of LSTM RNNs[25] and train them by connectionist temporal classification (CTC)[5] to find an RNN weight matrix that maximizes the probability of the label sequences in a training set, given the corresponding input sequences. CTC achieves both alignment and recognition.
Alternatives
Sometimes, it can be advantageous to train (parts of) an LSTM by neuroevolution[7] or by policy gradient methods, especially when there is no "teacher" (that is, training labels). 



"""

result = chain.invoke({'text': text})

print(result)

chain.get_graph().print_ascii()

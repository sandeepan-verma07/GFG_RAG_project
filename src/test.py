## GUYS put ur promt in the question ="..." add ur pdf make sure it has text in the data folder.

from rag_core import rag_answer

if __name__ == "__main__":
    question = " what is the methodology used in the research paper ?"
    answer = rag_answer(question)
    print("\nANSWER:\n", answer)
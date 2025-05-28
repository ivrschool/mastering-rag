from llm.model_factory import get_model
import os

def main():
    llm = get_model()
    prompt = input("Enter your prompt: ")
    response = llm.generate_response(prompt)
    print("Response:", response)

if __name__ == "__main__":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    main()

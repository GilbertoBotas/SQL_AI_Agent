from modules.sql_agent import SQLAgent

def main():
    print("Welcome to the personal SQL Agent!")
    print("Initializing...")
    
    agent = SQLAgent()
    try:
        agent.initialize()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return

    print("\nAgent ready! Type 'exit' to quit.")
    while True:
        query = input("\nHow can I help you? > ")
        if query.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        
        if not query.strip():
            continue

        print("Searching and generating answer...")
        try:
            answer = agent.ask(query)
            print(f"\nAnswer: {answer.model_dump_json(indent = 2)}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
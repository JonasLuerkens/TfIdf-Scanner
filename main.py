from data_import import select_corpus
from scanner import Scanner

def main():
    print("TF-IDF-Scanner: (write 'exit' to quit at any time)")
   
    # Load corpus data
    docs, ids, corpus_name = select_corpus()
    print(f"Current corpus: {corpus_name}")

    # Init Scanner and training it on corps data
    scanner = Scanner()
    scanner.train(docs, ids)
    
    # Interaction loop
    while True:
        print("---------------------------------------------------")
        print("Enter your query:")
        user_input = input("Query: ")
        
        if user_input.lower() == 'exit':
            break
        
        # Handling empty input (query)
        if not user_input.strip():
            continue
        
        # Percentile threshold input
        percentile_input = input("State your percentile threshold (0-100): ")

        if percentile_input.lower() == 'exit':
            break
        
        # Handling empty input (perzentile)
        try:
            percentile = float(percentile_input)
        except ValueError:
            print("Error: Please enter a valid number for the percentile.")
            continue

        # Search for similar documents
        candidates, _ = scanner.find_similars(user_input, percentile)
        
        print(f"Found {len(candidates)} documents: \n")

        # Display output
        for i, res in enumerate(candidates[:10]):
            print(f"{i+1}. ID: {res['id']} | Similarity: {res['score']:.4f}")
            print(f"Preview: {res['preview'].replace('\n', ' ')}...\n")

        # More than 10 results
        if len(candidates) > 10:
            print(f"... and {len(candidates) - 10} more.")
            
if __name__ == "__main__":
    main()
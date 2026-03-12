
import json
import random
from pathlib import Path

def generate_scientific_document(id):
    topics = ["Quantum Computing", "Cancer Research", "Artificial Intelligence", "Genetics", "Astrophysics", "Neuroscience", "Nanotechnology", "Climate Science", "Molecular Biology", "Theoretical Physics"]
    actions = ["investigated", "analyzed", "simulated", "observed", "synthesized", "modeled", "discovered", "optimized", "characterized", "explored"]
    subjects = ["protein folding", "qubit coherence", "neural network convergence", "gene expression", "black hole radiation", "synaptic plasticity", "carbon nanotubes", "ocean acidification", "enzyme kinetics", "dark matter distribution"]
    results = ["significant improvement", "unexpected anomalies", "linear correlation", "exponential decay", "quantum superposition", "high fidelity", "robust performance", "enhanced stability", "novel phase transitions", "improved efficiency"]
    
    topic = random.choice(topics)
    action = random.choice(actions)
    subject = random.choice(subjects)
    result = random.choice(results)
    
    sentences = [
        f"In this study on {topic}, we {action} the properties of {subject}.",
        f"The experimental results demonstrated a {result} across all test cases.",
        f"Furthermore, the interaction between {subject} and external fields was examined.",
        f"Our findings suggest that {topic} principles can be applied to optimize {subject}.",
        f"This leads to a {result} in the overall system performance.",
        f"Future research should focus on the long-term effects of {subject} in various environments.",
        f"We conclude that {topic} remains a critical field for understanding {subject}.",
        f"The data indicates a clear {result} when using our proposed methodology."
    ]
    
    random.shuffle(sentences)
    text = " ".join(sentences)
    
    return {
        "id": id,
        "topic": topic,
        "text": text
    }

def main():
    num_docs = 10000
    print(f"Generating {num_docs} scientific documents...")
    
    data = []
    for i in range(num_docs):
        data.append(generate_scientific_document(i))
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1} documents")
            
    with open("massive_training_data.json", "w") as f:
        json.dump(data, f)
        
    print(f"✅ Successfully generated massive_training_data.json ({len(data)} documents)")

if __name__ == "__main__":
    main()

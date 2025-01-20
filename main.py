# Importiamo il modello BGEM3FlagModel dal modulo FlagEmbedding
from FlagEmbedding import BGEM3FlagModel

# Importiamo NumPy per gestire gli array e calcolare la cosine similarity
import numpy as np

# Importiamo TSNE da scikit-learn per ridurre la dimensionalità degli embeddings
from sklearn.manifold import TSNE

# Importiamo cosine_similarity per calcolare la similarità coseno
from sklearn.metrics.pairwise import cosine_similarity

# Importiamo matplotlib per visualizzare i risultati in un grafico 2D
import matplotlib.pyplot as plt

# Inizializziamo il modello BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Definiamo un elenco di frasi per calcolare gli embeddings
frasi = [
    # Argomento 1: Cibo
    "La pizza napoletana è uno dei simboli più famosi dell'Italia nel mondo.",
    "Il gelato artigianale italiano è considerato uno dei migliori al mondo.",
    "La pasta è un piatto tradizionale italiano, apprezzato in tutto il mondo.",
    "La mozzarella di bufala è un'eccellenza del sud Italia.",
    "Il tiramisù è uno dei dessert italiani più conosciuti e amati.",
    
    # Argomento 2: Automobili e motori
    "La Ferrari è un'icona del settore automobilistico italiano e sinonimo di velocità.",
    "La corsa di Formula 1 a Monza è un evento imperdibile per gli appassionati di motori.",
    "Il design delle automobili italiane è famoso per eleganza e prestazioni.",
    "La Lamborghini è un simbolo di lusso e potenza tra le supercar.",
    "L'Alfa Romeo ha una lunga storia nelle corse automobilistiche.",
    "La SF90 Stradale è dotata di un motore endotermico turbo a V di 90° in grado di erogare 780 cv, la potenza più alta mai raggiunta da un 8 cilindri nella storia della Ferrari"
]

# Definiamo una query per calcolare il suo embedding
query = "Quanti cavalli ha la ferrari SF90 stradale?"

# Calcoliamo gli embeddings per le frasi e per la query
embeddings = model.encode(frasi)['dense_vecs']
query_embedding = model.encode([query])['dense_vecs']

# Calcoliamo la cosine similarity tra la query e tutte le frasi
similarity_scores = cosine_similarity(query_embedding, embeddings)[0]

# Ordiniamo le frasi in base alla similarità con la query (in ordine decrescente)
sorted_indices = np.argsort(similarity_scores)[::-1]
sorted_results = [(frasi[i], similarity_scores[i]) for i in sorted_indices]

# Stampiamo i risultati ordinati
print("Risultati ordinati per similarità coseno con la query:")

for frase, score in sorted_results:
    print(f"Similarità: {score:.4f} - Frase: {frase}")

# Aggiungiamo l'embedding della query agli embeddings per la riduzione con TSNE
all_embeddings = np.vstack([embeddings, query_embedding])

# Riduciamo la dimensionalità degli embeddings a 2D con TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=10)
all_embeddings_2d = tsne.fit_transform(all_embeddings)

# Separiamo gli embeddings ridotti
embeddings_2d = all_embeddings_2d[:-1]  # Embeddings delle frasi
query_2d = all_embeddings_2d[-1]       # Embedding della query

# Creiamo il grafico 2D degli embeddings ridotti
plt.figure(figsize=(10, 8))

# Plottiamo i punti 2D delle frasi
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', label='Frasi')

# Annotiamo ogni punto con la frase corrispondente
for i, frase in enumerate(frasi):
    plt.text(embeddings_2d[i, 0] + 0.2, embeddings_2d[i, 1] + 0.2, frase, fontsize=8)

# Plottiamo la query come un punto rosso
plt.scatter(query_2d[0], query_2d[1], c='red', label='Query', s=100)
plt.text(query_2d[0] + 0.2, query_2d[1] + 0.2, "Query: " + query, fontsize=10, color='red')

# Aggiungiamo titolo, etichette e legenda
plt.title('Visualizzazione degli Embeddings con t-SNE')
plt.xlabel('Dimensione 1')
plt.ylabel('Dimensione 2')
plt.legend()
plt.grid(True)

# Mostriamo il grafico
plt.show()


## **Simple RAG System Python**
Un sistema di **Retrieval-Augmented Generation (RAG)** basato su Python, progettato per utilizzare il modello **BGE-M3** per calcolare embeddings semantici e implementare un workflow completo per il recupero delle informazioni e la loro analisi.

---

### **Caratteristiche principali**
- Utilizzo del modello di embeddings **BGE-M3** (https://huggingface.co/BAAI/bge-m3).
- Calcolo della **cosine similarity** per misurare la somiglianza semantica tra frasi e query.
- Visualizzazione grafica delle relazioni semantiche tramite **t-SNE**.
- Workflow completo di un sistema RAG:
  1. **Retrieval**: Recupero delle frasi più rilevanti rispetto alla query.
  2. **Augmentation**: Contestualizzazione dei dati recuperati.
  3. **Generation** (non implementata ma integrabile): Produzione di risposte basate sui dati recuperati.

---

### **Requisiti**
- Python 3.8 o superiore.
- Librerie Python necessarie:
  - **FlagEmbedding**: Per calcolare embeddings.
  - **NumPy**: Per manipolazione di array.
  - **scikit-learn**: Per cosine similarity e TSNE.
  - **Matplotlib**: Per la visualizzazione dei dati.

Installa le librerie con:
```bash
pip install FlagEmbedding numpy scikit-learn matplotlib
```

---

### **Modello utilizzato**
**BGE-M3** è un modello di embeddings pre-addestrato progettato per rappresentare frasi e testi come vettori numerici, ottimizzati per il recupero di informazioni e la ricerca semantica. È stato addestrato su grandi dataset per catturare relazioni semantiche complesse.

- **Fonte modello**: [Hugging Face - BGE-M3](https://huggingface.co/BAAI/bge-m3)
- **Descrizione dettagliata del modello**: [Guida agli embedding open-source](https://bentoml.com/blog/a-guide-to-open-source-embedding-models)

---

### **Come funziona il progetto?**

1. **Definizione delle frasi e della query:**
   - Il progetto prende in input un elenco di frasi (database) e una query dell’utente.

2. **Calcolo degli embeddings:**
   - Le frasi e la query vengono trasformate in vettori numerici ad alta dimensione utilizzando il modello BGE-M3.

3. **Calcolo della similarità:**
   - La similarità semantica tra la query e ciascuna frase viene misurata usando la cosine similarity.

4. **Ordinamento dei risultati:**
   - Le frasi più correlate alla query vengono ordinate in base al punteggio di similarità.

5. **Visualizzazione grafica:**
   - Gli embeddings vengono ridotti a 2 dimensioni con t-SNE e visualizzati in un grafico per mostrare le relazioni semantiche.

---

### **Come eseguire il progetto**

1. Clona la repository:
   ```bash
   git clone https://github.com/tuo-utente/simple-rag-system-python.git
   cd simple-rag-system-python
   ```

2. Installa i requisiti:
   ```bash
   pip install -r requirements.txt
   ```

3. Esegui lo script principale:
   ```bash
   python3 main.py
   ```

4. Analizza i risultati:
   - Le frasi più simili alla query saranno mostrate in ordine decrescente nella console.
   - Un grafico 2D mostrerà le relazioni tra frasi e query.

---

### **Esempio di output**
#### **Console:**
```
Risultati ordinati per similarità coseno con la query:
Similarità: 0.9432 - Frase: La SF90 Stradale è dotata di un motore endotermico turbo a V di 90°...
Similarità: 0.8315 - Frase: La Ferrari è un'icona del settore automobilistico italiano e sinonimo di velocità.
Similarità: 0.3123 - Frase: La Lamborghini è un simbolo di lusso e potenza tra le supercar.
...
```

#### **Grafico:**
- **Punti blu**: Frasi del database.
- **Punto rosso**: Query dell’utente.
- Le frasi correlate alla query saranno vicine nel grafico.

---

### **Possibili applicazioni**
- **Sistemi di FAQ**: Rispondere a domande frequenti basandosi su un database di conoscenze.
- **Ricerca semantica**: Recuperare documenti o frasi pertinenti rispetto a una query.
- **Analisi dei dati**: Visualizzare relazioni semantiche tra frasi in un dataset.

---

### **Espansioni future**
- Integrazione di un modello generativo per completare il ciclo RAG.
- Possibilità di caricare database personalizzati direttamente da file.
- Miglioramento della scalabilità per database di grandi dimensioni.

---


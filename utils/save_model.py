import torch
from sentence_transformers import SentenceTransformer

# Cargar tu modelo
model = SentenceTransformer('models/sentence-transformer-fact-check')

# Guardar el modelo como archivo torch
torch.save(model.state_dict(), 'modelo_sentence_transformer.pt')

# Para cargarlo después:
nuevo_modelo = SentenceTransformer('models/sentence-transformer-fact-check')
nuevo_modelo.load_state_dict(torch.load('modelo_sentence_transformer.pt'))
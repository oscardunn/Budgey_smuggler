# Assuming you've defined a model 'model_tuned'
num_layers = len(model_tuned.layers)

print(f"The model has {num_layers} layers.")

for i, layer in enumerate(model_tuned.layers):
    print(f"Layer {i + 1}: {layer.name}, Type: {type(layer).__name__}")
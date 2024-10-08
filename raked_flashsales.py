import tensorflow as tf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
import numpy as np

# Configuração do TensorFlow para uso de GPU
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Configuração do PyTorch para uso de GPU e computação distribuída
def setup(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# Função para carregar e preprocessar os dados
def load_data():
    # Exemplo de dados 
    data = {
        'horario': ['6-8', '8-10', '10-12', '12-14', '14-16', '16-18', '18-20', '20-22', '22-6'],
        'produto': ['Roupas', 'Ferramentas', 'Produtos de academia', 'Produtos de carro', 'Moveis'],
        'sexo': ['M', 'F'],
        'idade': [18, 25, 35, 45, 55],
        'estado_civil': ['Solteiro', 'Casado'],
        'emprego': ['Empregado', 'Desempregado']
    }
    df = pd.DataFrame(data)
    return df

# Função para treinar o modelo com TensorFlow
def train_tensorflow_model(data):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(len(data.columns),)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(data, epochs=10)
    return model

# Função para treinar o modelo com PyTorch
def train_pytorch_model(rank, world_size, data):
    setup(rank, world_size)
    model = torch.nn.Sequential(
        torch.nn.Linear(len(data.columns), 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
        torch.nn.Sigmoid()
    ).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    for epoch in range(10):
        for batch in data:
            optimizer.zero_grad()
            outputs = ddp_model(batch)
            loss = criterion(outputs, batch['label'])
            loss.backward()
            optimizer.step()
    cleanup()
    return ddp_model

# Função principal
def main():
    data = load_data()
    # Dividir os dados para treinamento distribuído
    world_size = 2
    torch.multiprocessing.spawn(train_pytorch_model, args=(world_size, data), nprocs=world_size, join=True)
    tensorflow_model = train_tensorflow_model(data)
    print("Modelos treinados com sucesso!")

if __name__ == "__main__":
    main()

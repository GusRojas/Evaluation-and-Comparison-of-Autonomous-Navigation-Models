#!/usr/bin/env python3
"""
Script para convertir modelos preentrenados de PyTorch (.pt) a formato ONNX
para despliegue en Raspberry Pi 5

Arquitecturas soportadas:
1. ConvLSTMModel
2. DroneConvMLP
3. DroneMLP
4. PilotNetRegressor
5. DroneMobileNetV3 (small)
6. DroneMobileNetV3 (large)
7. DroneResNet
8. DroneNavSAConvLSTM (use_sam=True, lightweight=True)
9. DroneNavSAConvLSTM (use_sam=True, lightweight=False)
10. DroneNavSAConvLSTM (use_sam=False, lightweight=False)
"""

import torch
import torch.onnx
import os
import sys
from pathlib import Path
import argparse

# Verificar dependencias críticas al inicio
def check_dependencies():
    """Verifica que las dependencias estén instaladas"""
    missing = []
    
    try:
        import onnx
    except ImportError:
        missing.append('onnx')
    
    try:
        import onnxscript
    except ImportError:
        missing.append('onnxscript')
    
    if missing:
        print("="*60)
        print("ERROR: DEPENDENCIAS FALTANTES")
        print("="*60)
        print(f"\nLos siguientes paquetes no están instalados:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPara instalarlos, ejecuta:")
        print("  pip install onnx onnxscript")
        print("\nO usa el script de instalación:")
        print("  bash install_dependencies.sh")
        print("="*60)
        return False
    
    return True

# Importar arquitecturas (asumiendo que arquitecturas.py está en el mismo directorio)
from arquitecturas import (
    ConvLSTMModel,
    DroneConvMLP,
    DroneMLP,
    PilotNetRegressor,
    DroneMobileNetV3,
    DroneResNet,
    DroneNavSAConvLSTM
)


# Configuraciones de modelos
MODEL_CONFIGS = {
    'convlstm': {
        'class': ConvLSTMModel,
        'params': {
            'input_channels': 1,
            'imu_dim': 6,
            'output_dim': 6,
            'hidden_dim': 32
        },
        'input_shapes': {
            'x_img': (1, 10, 1, 128, 128),  # [batch, seq_len, channels, H, W]
            'x_imu': (1, 10, 6)              # [batch, seq_len, imu_features]
        },
        'checkpoint': 'ConvLSTM.pt',
        'onnx_name': 'ConvLSTM.onnx'
    },
    
    'droneconvmlp': {
        'class': DroneConvMLP,
        'params': {
            'image_size': (128, 128),
            'num_frames': 10,
            'imu_features': 6,
            'output_size': 6,
            'img_channels': 1
        },
        'input_shapes': {
            'img': (1, 10, 1, 128, 128),
            'imu': (1, 10, 6)
        },
        'checkpoint': 'ConvMLP.pt',
        'onnx_name': 'ConvMLP.onnx'
    },
    
    'dronemlp': {
        'class': DroneMLP,
        'params': {},
        'input_shapes': {
            'x_img': (1, 10, 1, 128, 128),
            'x_imu': (1, 10, 6)
        },
        'checkpoint': 'MLP.pt',
        'onnx_name': 'droneMLP.onnx'
    },
    
    'pilotnet': {
        'class': PilotNetRegressor,
        'params': {
            'image_size': (128, 128),
            'sequence_length': 10,
            'imu_features': 6,
            'output_size': 6
        },
        'input_shapes': {
            'image_seq': (1, 10, 1, 128, 128),
            'imu_seq': (1, 10, 6)
        },
        'checkpoint': 'DronePilotNetRegressor.pt',
        'onnx_name': 'PilotNetRegressor.onnx'
    },
    
    'mobilenetv3_small': {
        'class': DroneMobileNetV3,
        'params': {
            'mobilenet_type': 'small',
            'weights': None
        },
        'input_shapes': {
            'x_img': (1, 10, 1, 128, 128),
            'x_imu': (1, 10, 6)
        },
        'checkpoint': 'DroneMobileNetv3_small.pt',
        'onnx_name': 'DroneMobileNetv3_small.onnx'
    },
    
    'mobilenetv3_large': {
        'class': DroneMobileNetV3,
        'params': {
            'mobilenet_type': 'large',
            'weights': None
        },
        'input_shapes': {
            'x_img': (1, 10, 1, 128, 128),
            'x_imu': (1, 10, 6)
        },
        'checkpoint': 'DroneMobileNetV3_large.pt',
        'onnx_name': 'DroneMobileNetV3_large.onnx'
    },
    
    'resnet': {
        'class': DroneResNet,
        'params': {
            'resnet_type': 'resnet18',
            'pretrained': False
        },
        'input_shapes': {
            'x_img': (1, 10, 1, 128, 128),
            'x_imu': (1, 10, 6)
        },
        'checkpoint': 'DroneResNet18.pt',
        'onnx_name': 'DroneResNet18.onnx'
    },
    
    'saconvlstm_sam_light': {
        'class': DroneNavSAConvLSTM,
        'params': {
            'image_channels': 1,
            'imu_channels': 6,
            'num_layers': 4,
            'hidden_channels': 64,
            'kernel_size': 3,
            'use_sam': True,
            'lightweight': True,
            'output_dim': 6
        },
        'input_shapes': {
            'image_seq': (1, 10, 1, 128, 128),
            'imu_seq': (1, 10, 6)
        },
        'checkpoint': 'DroneNavSA-ConvLSTM_ligero.pt',
        'onnx_name': 'DroneNavSA-ConvLSTM_ligero.onnx'
    },
    
    'saconvlstm_sam_full': {
        'class': DroneNavSAConvLSTM,
        'params': {
            'image_channels': 1,
            'imu_channels': 6,
            'num_layers': 4,
            'hidden_channels': 64,
            'kernel_size': 3,
            'use_sam': True,
            'lightweight': False,
            'output_dim': 6
        },
        'input_shapes': {
            'image_seq': (1, 10, 1, 128, 128),
            'imu_seq': (1, 10, 6)
        },
        'checkpoint': 'DroneNavSA-ConvLSTM_completo.pt',
        'onnx_name': 'DroneNavSA-ConvLSTM_completo.onnx'
    },
    
    'saconvlstm_no_sam': {
        'class': DroneNavSAConvLSTM,
        'params': {
            'image_channels': 1,
            'imu_channels': 6,
            'num_layers': 4,
            'hidden_channels': 64,
            'kernel_size': 3,
            'use_sam': False,
            'lightweight': False,  # Este parámetro no se usa cuando use_sam=False
            'output_dim': 6
        },
        'input_shapes': {
            'image_seq': (1, 10, 1, 128, 128),
            'imu_seq': (1, 10, 6)
        },
        'checkpoint': 'DroneNav-ConvLSTM.pt',
        'onnx_name': 'DroneNav-ConvLSTM.onnx'
    }
}


def load_model(model_name, checkpoint_path, device='cpu'):
    """
    Carga un modelo desde un checkpoint
    
    Args:
        model_name: Nombre del modelo en MODEL_CONFIGS
        checkpoint_path: Ruta al archivo .pt
        device: Dispositivo para cargar el modelo ('cpu' o 'cuda')
    
    Returns:
        model: Modelo cargado en modo evaluación
    """
    print(f"\n{'='*60}")
    print(f"Cargando modelo: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el checkpoint: {checkpoint_path}")
    
    config = MODEL_CONFIGS[model_name]
    
    # Crear instancia del modelo
    model = config['class'](**config['params'])
    
    # Cargar pesos
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Manejar diferentes formatos de checkpoint
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  ✓ Cargado desde 'model_state_dict'")
            if 'epoch' in checkpoint:
                print(f"  ✓ Época del checkpoint: {checkpoint['epoch']}")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"  ✓ Cargado desde 'state_dict'")
        else:
            # Asumir que el dict es el state_dict directamente
            model.load_state_dict(checkpoint)
            print(f"  ✓ Cargado state_dict directamente")
    else:
        # El checkpoint es el modelo completo
        model = checkpoint
        print(f"  ✓ Cargado modelo completo")
    
    model.to(device)
    model.eval()
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Parámetros totales: {total_params:,}")
    print(f"  ✓ Parámetros entrenables: {trainable_params:,}")
    
    return model


def create_dummy_inputs(input_shapes, device='cpu'):
    """
    Crea inputs dummy para la exportación ONNX
    
    Args:
        input_shapes: Diccionario con las formas de las entradas
        device: Dispositivo donde crear los tensores
    
    Returns:
        tuple: Tupla con los tensores de entrada
    """
    inputs = []
    for name, shape in input_shapes.items():
        tensor = torch.randn(*shape).to(device)
        inputs.append(tensor)
        print(f"  ✓ Input '{name}': {list(shape)}")
    return tuple(inputs)


def convert_to_onnx(model, dummy_inputs, output_path, input_names, output_names, 
                    opset_version=14, dynamic_axes=None):
    """
    Convierte un modelo PyTorch a formato ONNX
    
    Args:
        model: Modelo PyTorch en modo evaluación
        dummy_inputs: Tupla con inputs de ejemplo
        output_path: Ruta donde guardar el archivo .onnx
        input_names: Lista con nombres de las entradas
        output_names: Lista con nombres de las salidas
        opset_version: Versión del opset de ONNX
        dynamic_axes: Diccionario con ejes dinámicos (opcional)
    """
    print(f"\nExportando a ONNX...")
    print(f"  Output: {output_path}")
    print(f"  Opset version: {opset_version}")
    
    # Configurar dynamic_axes por defecto si no se proporciona
    if dynamic_axes is None:
        dynamic_axes = {}
        for name in input_names:
            dynamic_axes[name] = {0: 'batch_size'}
        for name in output_names:
            dynamic_axes[name] = {0: 'batch_size'}
    
    try:
        torch.onnx.export(
            model,
            dummy_inputs,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False,
            dynamo=False
        )
        
        # Verificar tamaño del archivo
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  ✓ Exportación exitosa!")
        print(f"  ✓ Tamaño del archivo: {file_size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error en la exportación: {str(e)}")
        return False


def verify_onnx_model(onnx_path):
    """
    Verifica que el modelo ONNX sea válido
    
    Args:
        onnx_path: Ruta al archivo .onnx
    
    Returns:
        bool: True si es válido, False en caso contrario
    """
    try:
        import onnx
        
        print(f"\nVerificando modelo ONNX...")
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"  ✓ Modelo ONNX válido!")
        
        # Mostrar información del modelo
        print(f"\n  Información del modelo:")
        print(f"  - Entradas:")
        for input_tensor in model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                    for dim in input_tensor.type.tensor_type.shape.dim]
            print(f"    - {input_tensor.name}: {shape}")
        
        print(f"  - Salidas:")
        for output_tensor in model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                    for dim in output_tensor.type.tensor_type.shape.dim]
            print(f"    - {output_tensor.name}: {shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error en la verificación: {str(e)}")
        return False


def main(args):
    """
    Función principal para convertir modelos
    """
    print("="*60)
    print("CONVERSIÓN DE MODELOS PYTORCH A ONNX")
    print("="*60)
    
    # Verificar dependencias
    if not check_dependencies():
        sys.exit(1)
    
    # Crear directorio de salida si no existe
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDirectorio de salida: {output_dir}")
    
    # Directorio de checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    print(f"Directorio de checkpoints: {checkpoint_dir}")
    
    # Dispositivo
    device = 'cpu'  # Forzar CPU para compatibilidad con RPI5
    print(f"Dispositivo: {device}")
    
    # Modelos a convertir
    if args.models == 'all':
        models_to_convert = list(MODEL_CONFIGS.keys())
    else:
        models_to_convert = args.models.split(',')
    
    print(f"\nModelos a convertir: {len(models_to_convert)}")
    for model_name in models_to_convert:
        if model_name not in MODEL_CONFIGS:
            print(f"  ✗ Modelo desconocido: {model_name}")
            models_to_convert.remove(model_name)
    
    # Contadores
    success_count = 0
    failed_models = []
    
    # Convertir cada modelo
    for i, model_name in enumerate(models_to_convert, 1):
        print(f"\n{'='*60}")
        print(f"MODELO {i}/{len(models_to_convert)}: {model_name}")
        print(f"{'='*60}")
        
        config = MODEL_CONFIGS[model_name]
        
        try:
            # Construir ruta al checkpoint
            checkpoint_path = checkpoint_dir / config['checkpoint']
            
            # Cargar modelo
            model = load_model(model_name, checkpoint_path, device)
            
            # Crear inputs dummy
            print(f"\nCreando inputs dummy:")
            dummy_inputs = create_dummy_inputs(config['input_shapes'], device)
            
            # Definir nombres de entrada/salida
            input_names = list(config['input_shapes'].keys())
            output_names = ['output']
            
            # Ruta de salida ONNX
            onnx_path = output_dir / config['onnx_name']
            
            # Convertir a ONNX
            success = convert_to_onnx(
                model=model,
                dummy_inputs=dummy_inputs,
                output_path=str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                opset_version=args.opset_version
            )
            
            if success and args.verify:
                # Verificar modelo ONNX
                verify_onnx_model(str(onnx_path))
            
            if success:
                success_count += 1
                print(f"\n✓ {model_name} convertido exitosamente!")
            else:
                failed_models.append(model_name)
                print(f"\n✗ {model_name} falló en la conversión")
            
        except Exception as e:
            failed_models.append(model_name)
            print(f"\n✗ Error procesando {model_name}:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Resumen final
    print(f"\n{'='*60}")
    print("RESUMEN DE CONVERSIÓN")
    print(f"{'='*60}")
    print(f"Total de modelos: {len(models_to_convert)}")
    print(f"Exitosos: {success_count}")
    print(f"Fallidos: {len(failed_models)}")
    
    if failed_models:
        print(f"\nModelos que fallaron:")
        for model_name in failed_models:
            print(f"  - {model_name}")
    
    print(f"\n{'='*60}")
    print("¡Conversión completada!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convertir modelos PyTorch a ONNX para RPI5'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Directorio con los checkpoints .pt (default: ./checkpoints)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./onnx_models',
        help='Directorio donde guardar los modelos ONNX (default: ./onnx_models)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        default='all',
        help='Modelos a convertir, separados por comas o "all" (default: all)'
    )
    
    parser.add_argument(
        '--opset-version',
        type=int,
        default=14,
        help='Versión del opset de ONNX (default: 14)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verificar modelos ONNX después de la conversión'
    )
    
    args = parser.parse_args()
    main(args)

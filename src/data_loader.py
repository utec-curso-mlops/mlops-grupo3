import pandas as pd
import os
from pathlib import Path

# Importar configuración
try:
    from config import DATA_CONFIG, MERGE_CONFIG, PATH_CONFIG
    print("✅ Configuración cargada desde config.py")
except ImportError:
    print("⚠️  config.py no encontrado, usando configuración por defecto")
    # Configuración por defecto si no existe config.py
    DATA_CONFIG = {
        "train": {
            "clientes": "train_clientes_sample.csv",
            "requerimientos": "train_requerimientos_sample.csv"
        },
        "oot": {
            "clientes": "oot_clientes_sample.csv", 
            "requerimientos": "oot_requerimientos_sample.csv"
        }
    }
    MERGE_CONFIG = {
        "join_keys": ["ID_CORRELATIVO", "CODMES"],
        "join_type": "left",
        "target_column": "ATTRITION"
    }
    PATH_CONFIG = {
        "data_folder": "data",
        "input_subfolder": "in",
        "output_subfolder": "out"
    }

def get_data_path(subfolder="in"):
    """
    Obtiene la ruta relativa a la carpeta data/
    
    Args:
        subfolder (str): Subcarpeta dentro de data/ (default: "in")
    
    Returns:
        Path: Ruta a la carpeta de datos
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    data_path = project_root / PATH_CONFIG["data_folder"] / subfolder
    return data_path

def load_individual_dataset(filename, subfolder="in"):
    """
    Carga un dataset individual usando rutas relativas
    
    Args:
        filename (str): Nombre del archivo CSV
        subfolder (str): Subcarpeta donde está el archivo
    
    Returns:
        pd.DataFrame: Dataset cargado
    """
    data_path = get_data_path(subfolder)
    file_path = data_path / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"✅ Dataset '{filename}' cargado: {df.shape}")
    return df

def validate_join_keys(df1, df2, join_keys):
    """
    Valida que las columnas de join existan en ambos DataFrames
    
    Args:
        df1, df2 (pd.DataFrame): DataFrames a validar
        join_keys (list): Columnas para el join
        
    Returns:
        bool: True si todas las columnas existen
    """
    missing_in_df1 = [key for key in join_keys if key not in df1.columns]
    missing_in_df2 = [key for key in join_keys if key not in df2.columns]
    
    if missing_in_df1:
        print(f"❌ Columnas faltantes en clientes: {missing_in_df1}")
        print(f"🔍 Columnas disponibles en clientes: {list(df1.columns)}")
        return False
        
    if missing_in_df2:
        print(f"❌ Columnas faltantes en requerimientos: {missing_in_df2}")
        print(f"🔍 Columnas disponibles en requerimientos: {list(df2.columns)}")
        return False
        
    print(f"✅ Columnas de join validadas: {join_keys}")
    return True

def load_and_merge_datasets(dataset_type="train", 
                          custom_files=None, 
                          custom_join_keys=None,
                          join_type=None):
    """
    Carga y cruza las bases de datos de manera flexible
    
    Args:
        dataset_type (str): "train" o "oot"
        custom_files (dict): Archivos personalizados {"clientes": "file1.csv", "requerimientos": "file2.csv"}
        custom_join_keys (list): Columnas personalizadas para el join
        join_type (str): Tipo de join ("left", "inner", "outer")
    
    Returns:
        pd.DataFrame: Dataset combinado
    """
    print(f"🔄 Iniciando carga y cruce para dataset tipo: {dataset_type}")
    
    # Usar archivos personalizados o de configuración
    if custom_files:
        clientes_file = custom_files["clientes"]
        requerimientos_file = custom_files["requerimientos"]
        print(f"📂 Usando archivos personalizados")
    elif dataset_type in DATA_CONFIG:
        clientes_file = DATA_CONFIG[dataset_type]["clientes"]
        requerimientos_file = DATA_CONFIG[dataset_type]["requerimientos"]
        print(f"📂 Usando archivos de configuración para {dataset_type}")
    else:
        raise ValueError(f"dataset_type '{dataset_type}' no encontrado en configuración")
    
    # Cargar datasets individuales
    df_clientes = load_individual_dataset(clientes_file)
    df_requerimientos = load_individual_dataset(requerimientos_file)
    
    # Determinar columnas para el join
    join_keys = custom_join_keys or MERGE_CONFIG["join_keys"]
    
    # Validar que las columnas existen
    if not validate_join_keys(df_clientes, df_requerimientos, join_keys):
        raise ValueError("Error en validación de columnas de join")
    
    # Determinar tipo de join
    merge_type = join_type or MERGE_CONFIG["join_type"]
    
    # Realizar el cruce
    try:
        print(f"🔗 Realizando {merge_type} join por: {join_keys}")
        df_merged = pd.merge(
            df_clientes, 
            df_requerimientos, 
            on=join_keys, 
            how=merge_type
        )
        
        print(f"✅ Cruce completado exitosamente:")
        print(f"   📊 Clientes: {df_clientes.shape[0]} registros")
        print(f"   📊 Requerimientos: {df_requerimientos.shape[0]} registros")
        print(f"   📊 Resultado final: {df_merged.shape[0]} registros")
        print(f"   📊 Columnas totales: {df_merged.shape[1]}")
        
        return df_merged
        
    except Exception as e:
        print(f"❌ Error en el cruce: {e}")
        raise

def load_data(dataset_type="train", **kwargs):
    """
    Función principal que carga y cruza los datos
    
    Args:
        dataset_type (str): "train" para entrenamiento, "oot" para evaluación (test)
        **kwargs: Parámetros adicionales para personalizar el comportamiento
    
    Returns:
        pd.DataFrame: Dataset combinado listo para usar
            - Si dataset_type="train": datos de entrenamiento con target ATTRITION
            - Si dataset_type="oot": datos de evaluación SIN target (para predicción)
    """
    print(f"🚀 INICIANDO CARGA DE DATOS")
    print(f"📋 Tipo de dataset: {dataset_type}")
    if dataset_type == "train":
        print("🎯 Propósito: Entrenamiento del modelo")
    else:
        print("🎯 Propósito: Evaluación del modelo (Out of Time)")
    print("=" * 60)
    
    # Cargar y cruzar datos
    merged_data = load_and_merge_datasets(dataset_type, **kwargs)
    
    print("=" * 60)
    print("🎉 CARGA COMPLETADA EXITOSAMENTE")
    
    return merged_data

# Función de compatibilidad
def load_data_legacy(file_path):
    """Función legacy para mantener compatibilidad"""
    print("⚠️  Usando función legacy. Considera usar load_data() en su lugar.")
    df = pd.read_csv(file_path)
    return df
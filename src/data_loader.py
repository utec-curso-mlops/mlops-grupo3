import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pathlib import Path

# Importar configuración
try:
    from config import DATA_CONFIG, MERGE_CONFIG, SPLIT_CONFIG, PATH_CONFIG
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
    SPLIT_CONFIG = {
        "test_size": 0.2,
        "random_state": 42,
        "stratify": True
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

def split_train_test(data, 
                   target_column=None, 
                   test_size=None, 
                   random_state=None,
                   stratify=None):
    """
    Divide el dataset en train y test de manera flexible
    
    Args:
        data (pd.DataFrame): Dataset combinado
        target_column (str): Nombre de la columna objetivo
        test_size (float): Proporción para test
        random_state (int): Semilla para reproducibilidad
        stratify (bool): Si mantener proporción de clases
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"✂️  Iniciando división train/test")
    
    # Usar parámetros personalizados o de configuración
    target_col = target_column or MERGE_CONFIG["target_column"]
    test_sz = test_size or SPLIT_CONFIG["test_size"]
    rand_state = random_state or SPLIT_CONFIG["random_state"]
    do_stratify = stratify if stratify is not None else SPLIT_CONFIG["stratify"]
    
    # Verificar que existe la columna objetivo
    if target_col not in data.columns:
        print(f"❌ Columna objetivo '{target_col}' no encontrada")
        print(f"🔍 Columnas disponibles: {list(data.columns)}")
        raise ValueError(f"Columna objetivo '{target_col}' no encontrada en el dataset")
    
    # Separar características (X) y variable objetivo (y)
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Configurar stratify
    stratify_param = y if do_stratify else None
    
    print(f"🎯 Configuración de división:")
    print(f"   📊 Target: '{target_col}'")
    print(f"   📊 Test size: {test_sz}")
    print(f"   📊 Random state: {rand_state}")
    print(f"   📊 Stratify: {do_stratify}")
    
    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_sz, 
        random_state=rand_state,
        stratify=stratify_param
    )
    
    print(f"✅ División completada exitosamente:")
    print(f"   📈 Train: {X_train.shape[0]} registros ({X_train.shape[0]/len(data)*100:.1f}%)")
    print(f"   📊 Test: {X_test.shape[0]} registros ({X_test.shape[0]/len(data)*100:.1f}%)")
    print(f"   🎯 Distribución train: {y_train.value_counts().to_dict()}")
    print(f"   🎯 Distribución test: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test

def load_data(dataset_type="train", **kwargs):
    """
    Función principal que carga, cruza y divide los datos
    
    Args:
        dataset_type (str): "train" para entrenamiento, "oot" para aplicación
        **kwargs: Parámetros adicionales para personalizar el comportamiento
    
    Returns:
        Si dataset_type="train": tuple (X_train, X_test, y_train, y_test)
        Si dataset_type="oot": pd.DataFrame (datos para scoring)
    """
    print(f"🚀 INICIANDO PIPELINE DE CARGA DE DATOS")
    print(f"📋 Tipo de dataset: {dataset_type}")
    print("=" * 60)
    
    # Cargar y cruzar datos
    merged_data = load_and_merge_datasets(dataset_type, **kwargs)
    
    if dataset_type == "train":
        print("\n" + "=" * 60)
        # Para datos de entrenamiento: dividir en train/test
        result = split_train_test(merged_data, **kwargs)
        print("=" * 60)
        print("🎉 PIPELINE COMPLETADO EXITOSAMENTE")
        return result
    else:
        # Para datos OOT: devolver dataset completo para scoring
        print("=" * 60)
        print(f"✅ Datos OOT listos para scoring: {merged_data.shape}")
        print("🎉 PIPELINE COMPLETADO EXITOSAMENTE")
        return merged_data

# Función de compatibilidad
def load_data_legacy(file_path):
    """Función legacy para mantener compatibilidad"""
    print("⚠️  Usando función legacy. Considera usar load_data() en su lugar.")
    df = pd.read_csv(file_path)
    return df
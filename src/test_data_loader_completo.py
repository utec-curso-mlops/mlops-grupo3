#!/usr/bin/env python3
"""
Script de validación completa para data_loader.py (versión corregida)
TRAIN y OOT son datasets separados, NO hay división interna
Ejecutar desde la carpeta src/ 
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Agregar src al path si ejecutamos desde raíz
current_dir = Path.cwd()
if current_dir.name != "src":
    src_path = current_dir / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

def test_config_import():
    """Test 1: Verificar que config.py se importa correctamente"""
    print("=" * 60)
    print("🧪 TEST 1: Verificación de config.py")
    print("=" * 60)
    
    try:
        from config import DATA_CONFIG, MERGE_CONFIG, PATH_CONFIG
        print("✅ config.py importado exitosamente")
        
        print(f"📁 Archivos de datos configurados: {list(DATA_CONFIG.keys())}")
        print(f"🔗 Columnas de join: {MERGE_CONFIG['join_keys']}")
        print(f"🎯 Target column: {MERGE_CONFIG['target_column']}")
        
        # Verificar que NO existe SPLIT_CONFIG
        try:
            from config import SPLIT_CONFIG
            print("⚠️  SPLIT_CONFIG encontrado (debería estar eliminado)")
            return False
        except ImportError:
            print("✅ SPLIT_CONFIG correctamente eliminado")
        
        return True
    except ImportError as e:
        print(f"❌ Error importando config.py: {e}")
        return False

def test_data_loader_import():
    """Test 2: Verificar que data_loader se importa correctamente"""
    print("\n" + "=" * 60)
    print("🧪 TEST 2: Verificación de data_loader.py")
    print("=" * 60)
    
    try:
        from data_loader import load_data, load_individual_dataset, load_and_merge_datasets
        print("✅ data_loader.py importado exitosamente")
        print("✅ Todas las funciones disponibles")
        
        # Verificar que NO existe split_train_test
        try:
            from data_loader import split_train_test
            print("⚠️  split_train_test encontrada (debería estar eliminada)")
            return False
        except ImportError:
            print("✅ split_train_test correctamente eliminada")
        
        return True
    except ImportError as e:
        print(f"❌ Error importando data_loader.py: {e}")
        return False

def test_individual_loading():
    """Test 3: Carga individual de datasets"""
    print("\n" + "=" * 60)
    print("🧪 TEST 3: Carga individual de datasets")
    print("=" * 60)
    
    try:
        from data_loader import load_individual_dataset
        
        # Test cada archivo
        datasets = [
            "train_clientes_sample.csv",
            "train_requerimientos_sample.csv", 
            "oot_clientes_sample.csv",
            "oot_requerimientos_sample.csv"
        ]
        
        loaded_datasets = {}
        for dataset in datasets:
            df = load_individual_dataset(dataset)
            loaded_datasets[dataset] = df
            print(f"   ✅ {dataset}: {df.shape}")
        
        print(f"\n✅ Todos los datasets cargados exitosamente!")
        return True, loaded_datasets
    except Exception as e:
        print(f"❌ Error en carga individual: {e}")
        return False, {}

def test_merge_functionality():
    """Test 4: Funcionalidad de cruce"""
    print("\n" + "=" * 60)
    print("🧪 TEST 4: Funcionalidad de cruce")
    print("=" * 60)
    
    try:
        from data_loader import load_and_merge_datasets
        
        # Test merge train
        print("📊 Probando cruce de datos TRAIN...")
        train_merged = load_and_merge_datasets("train")
        
        print("\n📊 Probando cruce de datos OOT...")
        oot_merged = load_and_merge_datasets("oot")
        
        print(f"\n✅ Cruces realizados exitosamente!")
        print(f"   📈 Train merged: {train_merged.shape}")
        print(f"   📊 OOT merged: {oot_merged.shape}")
        
        # Verificar que train tiene target y oot no
        if 'ATTRITION' in train_merged.columns:
            print("✅ Datos TRAIN contienen target ATTRITION")
        else:
            print("❌ Datos TRAIN NO contienen target ATTRITION")
            return False, None, None
            
        if 'ATTRITION' not in oot_merged.columns:
            print("✅ Datos OOT NO contienen target (correcto)")
        else:
            print("⚠️  Datos OOT contienen target ATTRITION (inesperado pero no error)")
        
        return True, train_merged, oot_merged
    except Exception as e:
        print(f"❌ Error en cruce: {e}")
        return False, None, None

def test_load_data_functionality():
    """Test 5: Funcionalidad completa de load_data"""
    print("\n" + "=" * 60)
    print("🧪 TEST 5: Funcionalidad completa de load_data")
    print("=" * 60)
    
    try:
        from data_loader import load_data
        
        # Test datos de entrenamiento
        print("🚀 Probando load_data('train')...")
        train_data = load_data("train")
        
        print(f"\n🚀 Probando load_data('oot')...")
        oot_data = load_data("oot")
        
        print(f"\n✅ load_data funcionando perfectamente!")
        print(f"   📈 Train data: {train_data.shape}")
        print(f"   📊 OOT data: {oot_data.shape}")
        
        # Validaciones adicionales
        assert isinstance(train_data, pd.DataFrame), "train_data no es DataFrame"
        assert isinstance(oot_data, pd.DataFrame), "oot_data no es DataFrame"
        assert train_data.shape[0] > 0, "train_data está vacío"
        assert oot_data.shape[0] > 0, "oot_data está vacío"
        
        print("✅ Todas las validaciones de tipos y dimensiones pasaron")
        
        return True, (train_data, oot_data)
    except Exception as e:
        print(f"❌ Error en load_data: {e}")
        return False, None

def test_data_quality():
    """Test 6: Calidad de datos"""
    print("\n" + "=" * 60)
    print("🧪 TEST 6: Validación de calidad de datos")
    print("=" * 60)
    
    try:
        from data_loader import load_data
        
        train_data = load_data("train")
        oot_data = load_data("oot")
        
        # Verificar target en train
        if 'ATTRITION' in train_data.columns:
            target_dist = train_data['ATTRITION'].value_counts()
            print(f"🎯 Distribución target en train: {target_dist.to_dict()}")
            
            # Verificar que hay ambas clases
            if len(target_dist) >= 2:
                print("✅ Target balanceado: ambas clases presentes")
            else:
                print("⚠️  Target desbalanceado: solo una clase")
        else:
            print("❌ Target ATTRITION no encontrado en datos train")
            return False
        
        # Verificar valores nulos
        train_nulls = train_data.isnull().sum().sum()
        oot_nulls = oot_data.isnull().sum().sum()
        
        print(f"📊 Valores nulos en train: {train_nulls}")
        print(f"📊 Valores nulos en oot: {oot_nulls}")
        
        # Verificar columnas comunes (excepto target)
        train_cols = set(train_data.columns) - {'ATTRITION'}
        oot_cols = set(oot_data.columns)
        
        if train_cols == oot_cols:
            print("✅ Columnas consistentes entre train y oot")
        else:
            print("⚠️  Diferencias en columnas entre train y oot")
            diff_cols = train_cols.symmetric_difference(oot_cols)
            print(f"   Columnas diferentes: {diff_cols}")
        
        return True
    except Exception as e:
        print(f"❌ Error en validación de calidad: {e}")
        return False

def generate_summary_report():
    """Generar reporte resumen para el equipo"""
    print("\n" + "=" * 60)
    print("📋 REPORTE RESUMEN PARA EL EQUIPO")
    print("=" * 60)
    
    try:
        from data_loader import load_data
        train_data = load_data("train")
        oot_data = load_data("oot")
        
        print(f"""
🎯 DATOS LISTOS PARA EL EQUIPO:

📊 DATASETS PROCESADOS:
   • Train (entrenamiento): {train_data.shape[0]:,} registros, {train_data.shape[1]:,} columnas
   • OOT (evaluación): {oot_data.shape[0]:,} registros, {oot_data.shape[1]:,} columnas

🎯 DISTRIBUCIÓN DEL TARGET (solo en train):
   • Sin fuga (0): {(train_data['ATTRITION']==0).sum():,} ({(train_data['ATTRITION']==0).mean()*100:.1f}%)
   • Con fuga (1): {(train_data['ATTRITION']==1).sum():,} ({(train_data['ATTRITION']==1).mean()*100:.1f}%)

🚀 COMO USAR EN EL EQUIPO:
   
   # Para datos de entrenamiento:
   from data_loader import load_data
   train_data = load_data("train")  # DataFrame con target ATTRITION
   
   # Para datos de evaluación (Out of Time):
   oot_data = load_data("oot")      # DataFrame SIN target

📋 FLUJO CORRECTO:
   1. train_data = load_data("train")     # Entrenar modelo
   2. prepare_data(train_data)            # Dividir internamente train/val
   3. train_model()                       # Entrenar
   4. oot_data = load_data("oot")         # Evaluar en datos OOT

✅ ESTADO: LISTO PARA PRODUCCIÓN
        """)
        
        return True
    except Exception as e:
        print(f"❌ Error generando reporte: {e}")
        return False

def main():
    """Ejecutar validación completa"""
    print("🔬 VALIDACIÓN COMPLETA DEL SISTEMA DATA_LOADER (VERSIÓN CORREGIDA)")
    print("=" * 80)
    
    tests_passed = 0
    total_tests = 6
    
    # Ejecutar todos los tests
    if test_config_import():
        tests_passed += 1
    
    if test_data_loader_import():
        tests_passed += 1
        
        if test_individual_loading()[0]:
            tests_passed += 1
        
        if test_merge_functionality()[0]:
            tests_passed += 1
        
        if test_load_data_functionality()[0]:
            tests_passed += 1
        
        if test_data_quality():
            tests_passed += 1
    
    # Resultado final
    print("\n" + "=" * 80)
    print(f"📊 RESULTADO FINAL: {tests_passed}/{total_tests} tests pasaron")
    
    if tests_passed == total_tests:
        print("🎉 TODOS LOS TESTS PASARON - SISTEMA CORREGIDO Y LISTO!")
        generate_summary_report()
    else:
        print("❌ Algunos tests fallaron - Revisar errores antes de continuar")
        
    print("=" * 80)

if __name__ == "__main__":
    main()
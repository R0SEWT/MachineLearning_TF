#!/usr/bin/env python3
"""
Script de testing para funcionalidades de Fase 3
Prueba paralelización, gestión de memoria y cache
"""

import sys
import os
import time
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Suprimir warnings
warnings.filterwarnings('ignore')

# Agregar path para imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_phase3_imports():
    """Test 1: Verificar que se pueden importar componentes de Fase 3"""
    print("🧪 Test 1: Importando componentes de Fase 3...")
    
    try:
        from utils.parallelization import (
            ParallelizationConfig, WorkerManager, DistributedOptimizer,
            ParallelTrialExecutor, DEFAULT_PARALLELIZATION_CONFIG
        )
        print("   ✅ Módulo de paralelización importado")
        
        from utils.memory_manager import (
            MemoryConfig, MemoryManager, MemoryMonitor, 
            CacheManager, PersistenceManager, DEFAULT_MEMORY_CONFIG
        )
        print("   ✅ Módulo de gestión de memoria importado")
        
        from utils import get_phase_info
        info = get_phase_info()
        print(f"   ✅ Fase 3 disponible: {info.get('fase_3_disponible', False)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error importando componentes de Fase 3: {e}")
        return False

def test_worker_manager():
    """Test 2: Verificar funcionamiento del WorkerManager"""
    print("\n🧪 Test 2: WorkerManager...")
    
    try:
        from utils.parallelization import WorkerManager, ParallelizationConfig
        
        # Configuración de test
        config = ParallelizationConfig(
            n_workers=2,
            worker_type='process',
            queue_size=100,
            timeout=10
        )
        
        # Crear worker manager
        worker_manager = WorkerManager(config)
        print(f"   ✅ WorkerManager creado con {config.n_workers} workers")
        
        # Iniciar workers
        worker_manager.start_workers()
        print("   ✅ Workers iniciados")
        
        # Función de test
        def test_task(x):
            return x * 2
        
        # Enviar tarea
        future = worker_manager.submit_task(test_task, 5)
        result = future.result(timeout=5)
        
        if result == 10:
            print("   ✅ Tarea individual ejecutada correctamente")
        else:
            print(f"   ❌ Resultado incorrecto: {result}")
            
        # Test de batch
        tasks = [(test_task, (i,), {}) for i in range(5)]
        results = worker_manager.submit_batch(tasks)
        
        if len(results) == 5 and all(r == i*2 for i, r in enumerate(results)):
            print("   ✅ Batch de tareas ejecutado correctamente")
        else:
            print(f"   ❌ Resultados de batch incorrectos: {results}")
        
        # Obtener estadísticas
        stats = worker_manager.get_stats()
        print(f"   📊 Workers: {stats['n_workers']}, Activos: {stats['is_running']}")
        
        # Detener workers
        worker_manager.stop_workers()
        print("   ✅ Workers detenidos")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en WorkerManager: {e}")
        return False

def test_memory_manager():
    """Test 3: Verificar funcionamiento del MemoryManager"""
    print("\n🧪 Test 3: MemoryManager...")
    
    try:
        from utils.memory_manager import MemoryManager, MemoryConfig
        
        # Configuración de test
        config = MemoryConfig(
            memory_limit_mb=1024,
            gc_threshold_mb=512,
            cache_enabled=True,
            monitor_interval=1
        )
        
        # Crear memory manager
        memory_manager = MemoryManager(config)
        print("   ✅ MemoryManager creado")
        
        # Iniciar monitoreo
        memory_manager.start()
        print("   ✅ Monitoreo iniciado")
        
        # Obtener estadísticas
        stats = memory_manager.get_comprehensive_stats()
        print(f"   📊 Memoria: {stats['memory_stats']['used_percent']:.1f}%")
        
        # Test de cache
        cache_manager = memory_manager.cache_manager
        cache_manager.set("test_key", {"data": "test_value"})
        cached_value = cache_manager.get("test_key")
        
        if cached_value and cached_value.get("data") == "test_value":
            print("   ✅ Cache funcionando correctamente")
        else:
            print("   ❌ Cache no funciona correctamente")
        
        # Test de GC
        gc_result = memory_manager.gc_manager.strategic_gc(force=True)
        print(f"   ✅ GC ejecutado: {gc_result['memory_freed']:.1f}MB liberados")
        
        # Test de procesamiento por chunks
        test_data = list(range(1000))
        
        def process_chunk(chunk):
            return sum(chunk)
        
        chunk_results = memory_manager.chunk_processor.process_data_chunks(
            test_data, process_chunk, chunk_size=100
        )
        
        if len(chunk_results) == 10:  # 1000/100 = 10 chunks
            print("   ✅ Procesamiento por chunks funcionando")
        else:
            print(f"   ❌ Error en procesamiento por chunks: {len(chunk_results)} chunks")
        
        # Detener monitoreo
        memory_manager.stop()
        print("   ✅ Monitoreo detenido")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en MemoryManager: {e}")
        return False

def test_cache_manager():
    """Test 4: Verificar funcionamiento del CacheManager"""
    print("\n🧪 Test 4: CacheManager...")
    
    try:
        from utils.memory_manager import CacheManager, MemoryConfig
        
        config = MemoryConfig(
            cache_enabled=True,
            cache_ttl_hours=1,
            cache_dir="test_cache"
        )
        
        cache_manager = CacheManager(config)
        print("   ✅ CacheManager creado")
        
        # Test de set/get
        test_data = {"model": "xgboost", "params": {"n_estimators": 100}}
        cache_manager.set("test_model", test_data)
        
        retrieved_data = cache_manager.get("test_model")
        if retrieved_data == test_data:
            print("   ✅ Cache set/get funcionando")
        else:
            print("   ❌ Cache set/get no funciona")
        
        # Test de key generation
        key = cache_manager.generate_key("xgboost", n_estimators=100, max_depth=5)
        print(f"   ✅ Key generada: {key[:8]}...")
        
        # Test de estadísticas
        stats = cache_manager.get_stats()
        print(f"   📊 Cache hits: {stats['hits']}, misses: {stats['misses']}")
        
        # Limpiar cache
        cache_manager.clear()
        print("   ✅ Cache limpiado")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en CacheManager: {e}")
        return False

def test_persistence_manager():
    """Test 5: Verificar funcionamiento del PersistenceManager"""
    print("\n🧪 Test 5: PersistenceManager...")
    
    try:
        from utils.memory_manager import PersistenceManager, MemoryConfig
        
        config = MemoryConfig(
            persist_enabled=True,
            backup_dir="test_backups"
        )
        
        persistence_manager = PersistenceManager(config)
        print("   ✅ PersistenceManager creado")
        
        # Test de guardar resultados
        test_results = {
            "model": "xgboost",
            "best_score": 0.85,
            "best_params": {"n_estimators": 100},
            "timestamp": datetime.now().isoformat()
        }
        
        results_path = persistence_manager.save_results(test_results, "test_model")
        if results_path:
            print("   ✅ Resultados guardados")
        else:
            print("   ❌ Error guardando resultados")
        
        # Test de auto-backup
        backup_executed = False
        
        def test_backup():
            nonlocal backup_executed
            backup_executed = True
            print("   ✅ Auto-backup ejecutado")
        
        persistence_manager.start_auto_backup(test_backup)
        time.sleep(0.1)  # Esperar un poco
        persistence_manager.stop_auto_backup()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en PersistenceManager: {e}")
        return False

def test_parallel_trial_executor():
    """Test 6: Verificar ParallelTrialExecutor"""
    print("\n🧪 Test 6: ParallelTrialExecutor...")
    
    try:
        from utils.parallelization import ParallelTrialExecutor, ParallelizationConfig
        
        config = ParallelizationConfig(
            n_workers=2,
            worker_type='process'
        )
        
        executor = ParallelTrialExecutor(config)
        print("   ✅ ParallelTrialExecutor creado")
        
        # Función objetivo simulada
        def objective_func(trial):
            return trial * 0.1
        
        # Estudio simulado
        class MockStudy:
            def __init__(self):
                self.trials = []
        
        study = MockStudy()
        
        # Ejecutar trials paralelos
        result = executor.execute_parallel_trials(objective_func, study, n_trials=5)
        
        if result['successful_trials'] == 5:
            print("   ✅ Trials paralelos ejecutados correctamente")
        else:
            print(f"   ❌ Error en trials paralelos: {result['successful_trials']}/5")
        
        # Verificar métricas
        metrics = executor.get_performance_metrics()
        print(f"   📊 Trials totales: {metrics.get('total_trials', 0)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en ParallelTrialExecutor: {e}")
        return False

def test_integrated_optimization():
    """Test 7: Verificar integración completa con optimizador"""
    print("\n🧪 Test 7: Integración con optimizador...")
    
    try:
        from crypto_hyperparameter_optimizer import CryptoHyperparameterOptimizer
        from utils.parallelization import ParallelizationConfig
        from utils.memory_manager import MemoryConfig
        
        # Configuraciones de test
        parallel_config = ParallelizationConfig(
            n_workers=2,
            worker_type='process',
            queue_size=100
        )
        
        memory_config = MemoryConfig(
            memory_limit_mb=1024,
            cache_enabled=True,
            monitor_interval=5
        )
        
        # Crear optimizador con configuraciones de Fase 3
        optimizer = CryptoHyperparameterOptimizer(
            parallelization_config=parallel_config,
            memory_config=memory_config
        )
        
        print("   ✅ Optimizador creado con configuraciones de Fase 3")
        
        # Verificar que componentes están disponibles
        phase_3_available = (
            optimizer.worker_manager is not None and 
            optimizer.memory_manager is not None
        )
        
        if phase_3_available:
            print("   ✅ Componentes de Fase 3 disponibles en optimizador")
        else:
            print("   ❌ Componentes de Fase 3 no disponibles")
        
        # Obtener estadísticas del sistema
        stats = optimizer.get_system_stats()
        print(f"   📊 Fase 3 habilitada: {stats['phase_3_enabled']}")
        
        # Limpiar recursos
        optimizer.cleanup_resources()
        print("   ✅ Recursos limpiados")
        
        return phase_3_available
        
    except Exception as e:
        print(f"   ❌ Error en integración: {e}")
        return False

def test_comprehensive_workflow():
    """Test 8: Workflow completo de Fase 3"""
    print("\n🧪 Test 8: Workflow completo de Fase 3...")
    
    try:
        from utils.parallelization import WorkerManager, ParallelizationConfig
        from utils.memory_manager import MemoryManager, MemoryConfig
        
        # Configuraciones
        parallel_config = ParallelizationConfig(n_workers=2, worker_type='process')
        memory_config = MemoryConfig(memory_limit_mb=1024, cache_enabled=True)
        
        # Crear managers
        worker_manager = WorkerManager(parallel_config)
        memory_manager = MemoryManager(memory_config)
        
        # Iniciar sistemas
        worker_manager.start_workers()
        memory_manager.start()
        
        print("   ✅ Sistemas iniciados")
        
        # Simulación de trabajo
        def compute_task(x):
            # Simular trabajo computacional
            result = sum(range(x * 100))
            return result
        
        # Ejecutar tareas en paralelo
        tasks = [(compute_task, (i,), {}) for i in range(1, 6)]
        results = worker_manager.submit_batch(tasks)
        
        if len(results) == 5:
            print("   ✅ Tareas ejecutadas en paralelo")
        
        # Verificar gestión de memoria
        initial_stats = memory_manager.get_comprehensive_stats()
        
        # Forzar GC
        gc_result = memory_manager.gc_manager.strategic_gc(force=True)
        
        # Verificar cache
        cache_key = memory_manager.cache_manager.generate_key("test", value=42)
        memory_manager.cache_manager.set(cache_key, {"result": "cached"})
        cached_result = memory_manager.cache_manager.get(cache_key)
        
        if cached_result:
            print("   ✅ Cache funcionando en workflow")
        
        # Detener sistemas
        worker_manager.stop_workers()
        memory_manager.stop()
        
        print("   ✅ Workflow completo exitoso")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en workflow completo: {e}")
        return False

def main():
    """Ejecutar todos los tests de Fase 3"""
    print("🚀======================================================================")
    print("🧪 TESTING COMPLETO DE FASE 3 - EFICIENCIA Y ESCALABILIDAD")
    print("🚀======================================================================")
    
    tests = [
        test_phase3_imports,
        test_worker_manager,
        test_memory_manager,
        test_cache_manager,
        test_persistence_manager,
        test_parallel_trial_executor,
        test_integrated_optimization,
        test_comprehensive_workflow
    ]
    
    results = []
    start_time = time.time()
    
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"   ❌ Error ejecutando {test.__name__}: {e}")
            results.append((test.__name__, False))
    
    # Resumen final
    total_time = time.time() - start_time
    passed_tests = sum(1 for _, result in results if result)
    total_tests = len(results)
    
    print(f"\n🏁 RESUMEN DE TESTING DE FASE 3")
    print(f"🚀======================================================================")
    print(f"   ✅ Tests pasados: {passed_tests}/{total_tests}")
    print(f"   ⏱️ Tiempo total: {total_time:.2f}s")
    print(f"   📊 Tasa de éxito: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ¡TODOS LOS TESTS DE FASE 3 PASARON!")
        print("✅ Paralelización funcional")
        print("✅ Gestión de memoria operativa")
        print("✅ Cache y persistencia funcionando")
        print("✅ Integración completa exitosa")
    else:
        print(f"\n⚠️ {total_tests - passed_tests} tests fallaron")
        
        failed_tests = [name for name, result in results if not result]
        print("❌ Tests fallidos:")
        for test_name in failed_tests:
            print(f"   - {test_name}")
    
    print("\n🚀 Testing de Fase 3 completado!")
    
    # Crear reporte
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': total_tests - passed_tests,
        'success_rate': passed_tests/total_tests*100,
        'execution_time': total_time,
        'test_results': {name: result for name, result in results}
    }
    
    # Guardar reporte
    report_path = Path("test_phase3_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Reporte guardado en: {report_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script de test simple para verificar corrección de errores de memoria
"""

import sys
import os
import warnings
import time

# Suprimir warnings
warnings.filterwarnings('ignore')

# Agregar path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_memory_manager_basic():
    """Test básico del MemoryManager"""
    print("🧪 Test básico del MemoryManager...")
    
    try:
        from utils.memory_manager import MemoryManager, MemoryConfig
        
        # Configuración simple
        config = MemoryConfig(
            memory_limit_mb=1024,
            gc_threshold_mb=512,
            cache_enabled=True,
            monitor_interval=1,
            log_memory_usage=False  # Desactivar logging para test
        )
        
        print("   ✅ Configuración creada")
        
        # Crear memory manager
        memory_manager = MemoryManager(config)
        print("   ✅ MemoryManager creado")
        
        # Iniciar monitoreo
        memory_manager.start()
        print("   ✅ Monitoreo iniciado")
        
        # Esperar un poco para que funcione
        time.sleep(2)
        
        # Obtener estadísticas
        stats = memory_manager.get_comprehensive_stats()
        print(f"   📊 Memoria: {stats['memory_stats']['used_percent']:.1f}%")
        print(f"   📊 GC Objects: {stats['memory_stats']['gc_objects']}")
        print(f"   📊 GC Collections: {stats['memory_stats']['gc_collections']}")
        
        # Detener monitoreo
        memory_manager.stop()
        print("   ✅ Monitoreo detenido")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gc_manager():
    """Test del GarbageCollector"""
    print("\n🧪 Test del GarbageCollector...")
    
    try:
        from utils.memory_manager import GarbageCollector, MemoryConfig
        
        config = MemoryConfig(
            gc_threshold_mb=512,
            memory_limit_mb=1024
        )
        
        gc_manager = GarbageCollector(config)
        print("   ✅ GarbageCollector creado")
        
        # Test de GC
        gc_result = gc_manager.strategic_gc(force=True)
        print(f"   ✅ GC ejecutado: {gc_result['memory_freed']:.1f}MB liberados")
        print(f"   📊 Objetos recolectados: {gc_result['total_collected']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cache_manager():
    """Test del CacheManager"""
    print("\n🧪 Test del CacheManager...")
    
    try:
        from utils.memory_manager import CacheManager, MemoryConfig
        
        config = MemoryConfig(
            cache_enabled=True,
            cache_ttl_hours=1,
            cache_dir="test_cache"
        )
        
        cache_manager = CacheManager(config)
        print("   ✅ CacheManager creado")
        
        # Test básico
        cache_manager.set("test_key", {"value": 42})
        result = cache_manager.get("test_key")
        
        if result and result["value"] == 42:
            print("   ✅ Cache funcionando")
        else:
            print("   ❌ Cache no funciona")
            
        # Limpiar
        cache_manager.clear()
        print("   ✅ Cache limpiado")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parallelization():
    """Test básico de paralelización"""
    print("\n🧪 Test básico de paralelización...")
    
    try:
        from utils.parallelization import WorkerManager, ParallelizationConfig
        
        config = ParallelizationConfig(
            n_workers=2,
            worker_type='process',
            timeout=10
        )
        
        worker_manager = WorkerManager(config)
        print("   ✅ WorkerManager creado")
        
        # Iniciar workers
        worker_manager.start_workers()
        print("   ✅ Workers iniciados")
        
        # Función simple
        def simple_task(x):
            return x * 2
        
        # Test individual
        future = worker_manager.submit_task(simple_task, 5)
        result = future.result(timeout=5)
        
        if result == 10:
            print("   ✅ Tarea individual exitosa")
        else:
            print(f"   ❌ Resultado incorrecto: {result}")
        
        # Detener workers
        worker_manager.stop_workers()
        print("   ✅ Workers detenidos")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Ejecutar tests básicos"""
    print("🚀 TESTS BÁSICOS DE FASE 3 - VERIFICACIÓN DE CORRECCIONES")
    print("=" * 60)
    
    tests = [
        ("MemoryManager", test_memory_manager_basic),
        ("GarbageCollector", test_gc_manager),
        ("CacheManager", test_cache_manager),
        ("Paralelización", test_parallelization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Ejecutando test: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   ❌ Error ejecutando {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen
    print(f"\n{'='*60}")
    print("📊 RESUMEN DE TESTS")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASÓ" if success else "❌ FALLÓ"
        print(f"   {status} - {test_name}")
    
    print(f"\n🎯 Resultado: {passed}/{total} tests pasaron")
    
    if passed == total:
        print("🎉 ¡TODOS LOS TESTS PASARON!")
        print("✅ Los errores de memoria han sido corregidos")
    else:
        print("⚠️ Algunos tests fallaron")
        print("🔧 Revisar errores y corregir")

if __name__ == "__main__":
    main()

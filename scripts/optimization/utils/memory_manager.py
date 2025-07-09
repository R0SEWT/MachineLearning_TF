#!/usr/bin/env python3
"""
Sistema de gestión de memoria y cache para optimización de hiperparámetros
Incluye garbage collection estratégico, cache de resultados y persistencia
"""

import gc
import time
import pickle
import json
import os
import logging
import warnings
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import threading
from collections import OrderedDict

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Configuración para gestión de memoria"""
    
    # Configuración de memoria
    memory_limit_mb: int = 8192  # 8GB límite
    gc_threshold_mb: int = 6144  # 6GB para activar GC
    chunk_size_mb: int = 512  # 512MB por chunk
    
    # Configuración de cache
    cache_enabled: bool = True
    cache_dir: str = "cache"
    cache_size_limit_mb: int = 2048  # 2GB cache máximo
    cache_ttl_hours: int = 24  # Time to live
    
    # Configuración de persistencia
    persist_enabled: bool = True
    backup_dir: str = "backups"
    backup_interval_hours: int = 6
    auto_save_enabled: bool = True
    
    # Configuración de monitoring
    monitor_interval: int = 30  # segundos
    memory_alert_threshold: float = 0.85  # 85% de uso
    log_memory_usage: bool = True

class MemoryMonitor:
    """Monitor de uso de memoria en tiempo real"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.is_monitoring = False
        self.monitor_thread = None
        self.memory_history = []
        self.alerts = []
        
    def start_monitoring(self):
        """Iniciar monitoreo de memoria"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Monitor de memoria iniciado")
    
    def stop_monitoring(self):
        """Detener monitoreo de memoria"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        logger.info("Monitor de memoria detenido")
    
    def _monitor_loop(self):
        """Loop principal de monitoreo"""
        while self.is_monitoring:
            try:
                memory_stats = self._get_memory_stats()
                self.memory_history.append(memory_stats)
                
                # Mantener solo últimas 100 mediciones
                if len(self.memory_history) > 100:
                    self.memory_history = self.memory_history[-100:]
                
                # Verificar alertas
                self._check_memory_alerts(memory_stats)
                
                # Log si está habilitado
                if self.config.log_memory_usage:
                    self._log_memory_stats(memory_stats)
                
                time.sleep(self.config.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error en monitor de memoria: {e}")
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de memoria"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            process = psutil.Process()
            
            # Manejar gc.get_stats() de forma segura
            try:
                gc_stats = gc.get_stats()
                gc_collections = len(gc_stats) if gc_stats else 0
            except Exception:
                gc_collections = 0
            
            # Manejar gc.get_objects() de forma segura
            try:
                gc_objects_count = len(gc.get_objects())
            except Exception:
                gc_objects_count = 0
            
            return {
                'timestamp': time.time(),
                'total_mb': memory.total / (1024 * 1024),
                'available_mb': memory.available / (1024 * 1024),
                'used_mb': memory.used / (1024 * 1024),
                'used_percent': memory.percent,
                'process_mb': process.memory_info().rss / (1024 * 1024),
                'gc_collections': gc_collections,
                'gc_objects': gc_objects_count
            }
        except ImportError:
            # Fallback sin psutil
            try:
                gc_stats = gc.get_stats()
                gc_collections = len(gc_stats) if gc_stats else 0
            except Exception:
                gc_collections = 0
                
            try:
                gc_objects_count = len(gc.get_objects())
            except Exception:
                gc_objects_count = 0
                
            return {
                'timestamp': time.time(),
                'total_mb': self.config.memory_limit_mb,
                'available_mb': self.config.memory_limit_mb * 0.4,
                'used_mb': self.config.memory_limit_mb * 0.6,
                'used_percent': 60.0,
                'process_mb': 1024.0,
                'gc_collections': gc_collections,
                'gc_objects': gc_objects_count
            }
        except Exception as e:
            # Fallback completo en caso de error
            logger.error(f"Error obteniendo estadísticas de memoria: {e}")
            return {
                'timestamp': time.time(),
                'total_mb': self.config.memory_limit_mb,
                'available_mb': self.config.memory_limit_mb * 0.5,
                'used_mb': self.config.memory_limit_mb * 0.5,
                'used_percent': 50.0,
                'process_mb': 512.0,
                'gc_collections': 0,
                'gc_objects': 0
            }
    
    def _check_memory_alerts(self, memory_stats: Dict[str, Any]):
        """Verificar y generar alertas de memoria"""
        if memory_stats['used_percent'] > self.config.memory_alert_threshold * 100:
            alert = {
                'timestamp': time.time(),
                'type': 'memory_high',
                'message': f"Uso de memoria alto: {memory_stats['used_percent']:.1f}%",
                'memory_stats': memory_stats
            }
            self.alerts.append(alert)
            logger.warning(alert['message'])
    
    def _log_memory_stats(self, memory_stats: Dict[str, Any]):
        """Log estadísticas de memoria"""
        logger.info(f"Memory: {memory_stats['used_percent']:.1f}% "
                   f"({memory_stats['used_mb']:.0f}MB/{memory_stats['total_mb']:.0f}MB), "
                   f"Process: {memory_stats['process_mb']:.0f}MB, "
                   f"GC Objects: {memory_stats['gc_objects']}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas actuales"""
        return self._get_memory_stats()
    
    def get_memory_history(self) -> List[Dict[str, Any]]:
        """Obtener historial de memoria"""
        return self.memory_history.copy()
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Obtener alertas de memoria"""
        return self.alerts.copy()

class GarbageCollector:
    """Gestor de garbage collection estratégico"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.gc_history = []
        
    def strategic_gc(self, force: bool = False) -> Dict[str, Any]:
        """Ejecutar garbage collection estratégico"""
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        # Verificar si es necesario GC
        if not force and memory_before < self.config.gc_threshold_mb:
            return {
                'executed': False,
                'reason': 'memory_below_threshold',
                'memory_before': memory_before
            }
        
        # Ejecutar GC por generaciones
        collected = []
        for generation in [0, 1, 2]:
            try:
                collected_count = gc.collect(generation)
                collected.append(collected_count)
            except Exception as e:
                logger.error(f"Error en GC generación {generation}: {e}")
                collected.append(0)
        
        memory_after = self._get_memory_usage()
        duration = time.time() - start_time
        memory_freed = memory_before - memory_after
        
        gc_result = {
            'executed': True,
            'timestamp': time.time(),
            'duration': duration,
            'memory_before': memory_before,
            'memory_after': memory_after,
            'memory_freed': memory_freed,
            'objects_collected': collected,
            'total_collected': sum(collected)
        }
        
        self.gc_history.append(gc_result)
        
        logger.info(f"GC ejecutado: {memory_freed:.1f}MB liberados en {duration:.2f}s, "
                   f"{sum(collected)} objetos recolectados")
        
        return gc_result
    
    def _get_memory_usage(self) -> float:
        """Obtener uso de memoria en MB"""
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return 1024.0  # Fallback
        except Exception as e:
            logger.error(f"Error obteniendo uso de memoria: {e}")
            return 1024.0  # Fallback en caso de error
    
    def auto_gc_if_needed(self) -> Optional[Dict[str, Any]]:
        """Ejecutar GC automático si es necesario"""
        memory_usage = self._get_memory_usage()
        
        if memory_usage > self.config.gc_threshold_mb:
            return self.strategic_gc(force=True)
        
        return None
    
    def get_gc_history(self) -> List[Dict[str, Any]]:
        """Obtener historial de GC"""
        return self.gc_history.copy()

class DataChunkProcessor:
    """Procesador de datos por chunks para manejar datasets grandes"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.chunk_history = []
        
    def process_data_chunks(self, data: Any, process_func: Callable, 
                           chunk_size: Optional[int] = None) -> List[Any]:
        """Procesar datos por chunks"""
        if chunk_size is None:
            chunk_size = self._calculate_optimal_chunk_size(data)
        
        chunks = self._split_data_into_chunks(data, chunk_size)
        results = []
        
        logger.info(f"Procesando {len(chunks)} chunks de tamaño ~{chunk_size}")
        
        for i, chunk in enumerate(chunks):
            try:
                start_time = time.time()
                
                # Procesar chunk
                chunk_result = process_func(chunk)
                results.append(chunk_result)
                
                # Estadísticas del chunk
                duration = time.time() - start_time
                chunk_info = {
                    'chunk_id': i,
                    'chunk_size': len(chunk) if hasattr(chunk, '__len__') else chunk_size,
                    'duration': duration,
                    'timestamp': time.time()
                }
                self.chunk_history.append(chunk_info)
                
                # GC estratégico después de cada chunk
                if i % 5 == 0:  # Cada 5 chunks
                    gc.collect(0)  # GC ligero
                
                logger.debug(f"Chunk {i+1}/{len(chunks)} procesado en {duration:.2f}s")
                
            except Exception as e:
                logger.error(f"Error procesando chunk {i}: {e}")
                results.append(None)
        
        return results
    
    def _calculate_optimal_chunk_size(self, data: Any) -> int:
        """Calcular tamaño óptimo de chunk"""
        try:
            # Estimar memoria por elemento
            if hasattr(data, '__len__'):
                total_size = len(data)
                # Usar configuración de chunk size
                target_chunk_memory_mb = self.config.chunk_size_mb
                estimated_mb_per_element = target_chunk_memory_mb / 1000  # Estimación conservadora
                chunk_size = int(target_chunk_memory_mb / estimated_mb_per_element)
                return min(chunk_size, total_size)
            else:
                return 1000  # Default
        except:
            return 1000  # Fallback
    
    def _split_data_into_chunks(self, data: Any, chunk_size: int) -> List[Any]:
        """Dividir datos en chunks"""
        if hasattr(data, '__getitem__') and hasattr(data, '__len__'):
            # Lista, array, etc.
            chunks = []
            for i in range(0, len(data), chunk_size):
                chunks.append(data[i:i + chunk_size])
            return chunks
        else:
            # Tipo no soportado, devolver como un solo chunk
            return [data]
    
    def get_chunk_history(self) -> List[Dict[str, Any]]:
        """Obtener historial de procesamiento de chunks"""
        return self.chunk_history.copy()

class CacheManager:
    """Gestor de cache para resultados de optimización"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache en memoria (LRU)
        self.memory_cache: OrderedDict = OrderedDict()
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'disk_reads': 0,
            'disk_writes': 0
        }
        
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del cache"""
        if not self.config.cache_enabled:
            return None
        
        # Verificar cache en memoria
        if key in self.memory_cache:
            # Mover al final (LRU)
            value = self.memory_cache.pop(key)
            self.memory_cache[key] = value
            self.cache_stats['hits'] += 1
            return value['data']
        
        # Verificar cache en disco
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Verificar TTL
                if self._is_cache_valid(cached_data):
                    # Agregar a cache en memoria
                    self.memory_cache[key] = cached_data
                    self._evict_if_needed()
                    
                    self.cache_stats['hits'] += 1
                    self.cache_stats['disk_reads'] += 1
                    return cached_data['data']
                else:
                    # Cache expirado, eliminar
                    cache_file.unlink()
                    
            except Exception as e:
                logger.error(f"Error leyendo cache {key}: {e}")
        
        self.cache_stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Any, metadata: Optional[Dict] = None):
        """Guardar valor en cache"""
        if not self.config.cache_enabled:
            return
        
        cache_data = {
            'data': value,
            'timestamp': time.time(),
            'metadata': metadata or {},
            'ttl_hours': self.config.cache_ttl_hours
        }
        
        # Guardar en cache en memoria
        self.memory_cache[key] = cache_data
        self._evict_if_needed()
        
        # Guardar en disco
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            self.cache_stats['disk_writes'] += 1
        except Exception as e:
            logger.error(f"Error guardando cache {key}: {e}")
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generar clave de cache a partir de parámetros"""
        key_data = {'args': args, 'kwargs': kwargs}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def clear(self):
        """Limpiar cache"""
        self.memory_cache.clear()
        
        # Limpiar cache en disco
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.error(f"Error eliminando cache {cache_file}: {e}")
        
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'disk_reads': 0,
            'disk_writes': 0
        }
        
        logger.info("Cache limpiado")
    
    def _is_cache_valid(self, cache_data: Dict) -> bool:
        """Verificar si cache es válido (TTL)"""
        age_hours = (time.time() - cache_data['timestamp']) / 3600
        return age_hours < cache_data.get('ttl_hours', self.config.cache_ttl_hours)
    
    def _evict_if_needed(self):
        """Evictar elementos del cache si es necesario"""
        max_memory_items = 100  # Máximo elementos en memoria
        
        while len(self.memory_cache) > max_memory_items:
            # Eliminar elemento más antiguo (LRU)
            evicted_key, _ = self.memory_cache.popitem(last=False)
            self.cache_stats['evictions'] += 1
            logger.debug(f"Cache eviction: {evicted_key}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'cache_enabled': self.config.cache_enabled
        }

class PersistenceManager:
    """Gestor de persistencia para estudios y resultados"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.backup_dir = Path(config.backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        self.last_backup = None
        self.auto_backup_thread = None
        self.is_auto_backup_running = False
        
    def save_study(self, study: Any, study_name: str) -> str:
        """Guardar estudio Optuna"""
        if not self.config.persist_enabled:
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"study_{study_name}_{timestamp}.pkl"
        filepath = self.backup_dir / filename
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(study, f)
            
            logger.info(f"Estudio guardado: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error guardando estudio: {e}")
            return ""
    
    def load_study(self, filepath: str) -> Optional[Any]:
        """Cargar estudio Optuna"""
        try:
            with open(filepath, 'rb') as f:
                study = pickle.load(f)
            
            logger.info(f"Estudio cargado: {filepath}")
            return study
            
        except Exception as e:
            logger.error(f"Error cargando estudio: {e}")
            return None
    
    def save_results(self, results: Dict[str, Any], name: str) -> str:
        """Guardar resultados de optimización"""
        if not self.config.persist_enabled:
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{name}_{timestamp}.json"
        filepath = self.backup_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Resultados guardados: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error guardando resultados: {e}")
            return ""
    
    def start_auto_backup(self, backup_func: Callable):
        """Iniciar backup automático"""
        if not self.config.persist_enabled or self.is_auto_backup_running:
            return
        
        self.is_auto_backup_running = True
        self.auto_backup_thread = threading.Thread(
            target=self._auto_backup_loop, 
            args=(backup_func,)
        )
        self.auto_backup_thread.daemon = True
        self.auto_backup_thread.start()
        
        logger.info(f"Auto-backup iniciado (intervalo: {self.config.backup_interval_hours}h)")
    
    def stop_auto_backup(self):
        """Detener backup automático"""
        self.is_auto_backup_running = False
        if self.auto_backup_thread:
            self.auto_backup_thread.join(timeout=1)
        logger.info("Auto-backup detenido")
    
    def _auto_backup_loop(self, backup_func: Callable):
        """Loop de backup automático"""
        while self.is_auto_backup_running:
            try:
                time.sleep(self.config.backup_interval_hours * 3600)
                
                if self.is_auto_backup_running:
                    backup_func()
                    self.last_backup = time.time()
                    logger.info("Auto-backup ejecutado")
                    
            except Exception as e:
                logger.error(f"Error en auto-backup: {e}")
    
    def cleanup_old_backups(self, max_age_days: int = 30):
        """Limpiar backups antiguos"""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        
        for backup_file in self.backup_dir.glob("*"):
            try:
                if backup_file.stat().st_mtime < cutoff_time:
                    backup_file.unlink()
                    logger.debug(f"Backup antiguo eliminado: {backup_file}")
            except Exception as e:
                logger.error(f"Error eliminando backup {backup_file}: {e}")

class MemoryManager:
    """Gestor principal de memoria y cache"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.monitor = MemoryMonitor(config)
        self.gc_manager = GarbageCollector(config)
        self.chunk_processor = DataChunkProcessor(config)
        self.cache_manager = CacheManager(config)
        self.persistence_manager = PersistenceManager(config)
        
        self.is_active = False
        
    def start(self):
        """Iniciar gestor de memoria"""
        if self.is_active:
            return
        
        self.is_active = True
        self.monitor.start_monitoring()
        
        logger.info("MemoryManager iniciado")
    
    def stop(self):
        """Detener gestor de memoria"""
        if not self.is_active:
            return
        
        self.is_active = False
        self.monitor.stop_monitoring()
        self.persistence_manager.stop_auto_backup()
        
        logger.info("MemoryManager detenido")
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimizar uso de memoria"""
        start_time = time.time()
        
        # GC estratégico
        gc_result = self.gc_manager.strategic_gc(force=True)
        
        # Limpiar cache antiguo
        old_cache_files = []
        for cache_file in self.cache_manager.cache_dir.glob("*.pkl"):
            try:
                if cache_file.stat().st_mtime < time.time() - (24 * 3600):  # 1 día
                    cache_file.unlink()
                    old_cache_files.append(str(cache_file))
            except Exception as e:
                logger.error(f"Error limpiando cache {cache_file}: {e}")
        
        duration = time.time() - start_time
        
        result = {
            'duration': duration,
            'gc_result': gc_result,
            'cache_files_cleaned': len(old_cache_files),
            'memory_stats': self.monitor.get_current_stats()
        }
        
        logger.info(f"Optimización de memoria completada en {duration:.2f}s")
        return result
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas completas"""
        return {
            'memory_stats': self.monitor.get_current_stats(),
            'memory_history': self.monitor.get_memory_history()[-10:],  # Últimas 10
            'memory_alerts': self.monitor.get_alerts()[-5:],  # Últimas 5
            'gc_history': self.gc_manager.get_gc_history()[-5:],  # Últimos 5
            'chunk_history': self.chunk_processor.get_chunk_history()[-10:],  # Últimos 10
            'cache_stats': self.cache_manager.get_stats(),
            'config': self.config.__dict__,
            'is_active': self.is_active
        }

# Configuraciones predefinidas
DEFAULT_MEMORY_CONFIG = MemoryConfig(
    memory_limit_mb=8192,
    gc_threshold_mb=6144,
    chunk_size_mb=512,
    cache_enabled=True,
    cache_size_limit_mb=2048,
    cache_ttl_hours=24,
    persist_enabled=True,
    backup_interval_hours=6,
    monitor_interval=30,
    log_memory_usage=True
)

HIGH_PERFORMANCE_MEMORY_CONFIG = MemoryConfig(
    memory_limit_mb=16384,  # 16GB
    gc_threshold_mb=12288,  # 12GB
    chunk_size_mb=1024,     # 1GB chunks
    cache_enabled=True,
    cache_size_limit_mb=4096,  # 4GB cache
    cache_ttl_hours=48,
    persist_enabled=True,
    backup_interval_hours=3,
    monitor_interval=15,
    log_memory_usage=True
)

# Instancia global
MEMORY_MANAGER = MemoryManager(DEFAULT_MEMORY_CONFIG)

if __name__ == "__main__":
    # Test básico
    print("🧠 Sistema de Gestión de Memoria inicializado")
    print(f"   📊 Límite memoria: {DEFAULT_MEMORY_CONFIG.memory_limit_mb}MB")
    print(f"   🗄️ Cache habilitado: {DEFAULT_MEMORY_CONFIG.cache_enabled}")
    print(f"   💾 Persistencia habilitada: {DEFAULT_MEMORY_CONFIG.persist_enabled}")
    print(f"   🔄 GC threshold: {DEFAULT_MEMORY_CONFIG.gc_threshold_mb}MB")
    
    # Test de manager
    MEMORY_MANAGER.start()
    time.sleep(2)
    stats = MEMORY_MANAGER.get_comprehensive_stats()
    print(f"   📈 Memoria actual: {stats['memory_stats']['used_percent']:.1f}%")
    MEMORY_MANAGER.stop()

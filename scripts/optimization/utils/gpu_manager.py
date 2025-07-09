#!/usr/bin/env python3
"""
Gestor inteligente de GPU para optimización de hiperparámetros
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Tuple
import warnings

# Configurar logging
logger = logging.getLogger(__name__)

class GPUManager:
    """Gestor inteligente de configuración GPU/CPU"""
    
    def __init__(self):
        self.gpu_available = False
        self.gpu_info = {}
        self.cuda_available = False
        self._detect_hardware()
    
    def _detect_hardware(self):
        """Detectar hardware disponible"""
        try:
            # Verificar CUDA con PyTorch si está disponible
            try:
                import torch
                self.cuda_available = torch.cuda.is_available()
                if self.cuda_available:
                    self.gpu_info['cuda_version'] = torch.version.cuda
                    self.gpu_info['device_count'] = torch.cuda.device_count()
                    self.gpu_info['device_name'] = torch.cuda.get_device_name(0)
                    self.gpu_info['memory_total'] = torch.cuda.get_device_properties(0).total_memory
                    logger.info(f"CUDA detectado: {self.gpu_info['device_name']}")
            except ImportError:
                logger.warning("PyTorch no disponible para verificación CUDA")
            
            # Verificar disponibilidad de GPU para cada librería
            self._check_xgboost_gpu()
            self._check_lightgbm_gpu()
            self._check_catboost_gpu()
            
        except Exception as e:
            logger.error(f"Error detectando hardware: {e}")
            self.gpu_available = False
    
    def _check_xgboost_gpu(self) -> bool:
        """Verificar soporte GPU en XGBoost"""
        try:
            import xgboost as xgb
            # Test rápido para verificar GPU
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    # Crear modelo de prueba con GPU
                    model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=1)
                    self.gpu_info['xgboost_gpu'] = True
                    logger.info("XGBoost GPU disponible")
                    return True
                except Exception:
                    self.gpu_info['xgboost_gpu'] = False
                    logger.warning("XGBoost GPU no disponible")
                    return False
        except ImportError:
            logger.error("XGBoost no instalado")
            return False
    
    def _check_lightgbm_gpu(self) -> bool:
        """Verificar soporte GPU en LightGBM"""
        try:
            import lightgbm as lgb
            # Verificar si LightGBM fue compilado con GPU
            try:
                # Test básico
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = lgb.LGBMClassifier(device='gpu', n_estimators=1)
                    self.gpu_info['lightgbm_gpu'] = True
                    logger.info("LightGBM GPU disponible")
                    return True
            except Exception:
                self.gpu_info['lightgbm_gpu'] = False
                logger.warning("LightGBM GPU no disponible")
                return False
        except ImportError:
            logger.error("LightGBM no instalado")
            return False
    
    def _check_catboost_gpu(self) -> bool:
        """Verificar soporte GPU en CatBoost"""
        try:
            import catboost as cb
            # CatBoost generalmente soporta GPU si CUDA está disponible
            if self.cuda_available:
                self.gpu_info['catboost_gpu'] = True
                logger.info("CatBoost GPU disponible")
                return True
            else:
                self.gpu_info['catboost_gpu'] = False
                logger.warning("CatBoost GPU no disponible (sin CUDA)")
                return False
        except ImportError:
            logger.error("CatBoost no instalado")
            return False
    
    def get_xgboost_config(self, fallback_to_cpu: bool = True) -> Dict[str, Any]:
        """Obtener configuración óptima para XGBoost"""
        if self.gpu_info.get('xgboost_gpu', False):
            config = {
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'predictor': 'gpu_predictor'
            }
            logger.info("Usando configuración GPU para XGBoost")
        elif fallback_to_cpu:
            config = {
                'tree_method': 'hist',
                'n_jobs': -1
            }
            logger.info("Usando configuración CPU para XGBoost")
        else:
            raise RuntimeError("GPU no disponible para XGBoost y fallback deshabilitado")
        
        return config
    
    def get_lightgbm_config(self, fallback_to_cpu: bool = True) -> Dict[str, Any]:
        """Obtener configuración óptima para LightGBM"""
        if self.gpu_info.get('lightgbm_gpu', False):
            config = {
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            }
            logger.info("Usando configuración GPU para LightGBM")
        elif fallback_to_cpu:
            config = {
                'device': 'cpu',
                'n_jobs': -1
            }
            logger.info("Usando configuración CPU para LightGBM")
        else:
            raise RuntimeError("GPU no disponible para LightGBM y fallback deshabilitado")
        
        return config
    
    def get_catboost_config(self, fallback_to_cpu: bool = True) -> Dict[str, Any]:
        """Obtener configuración óptima para CatBoost"""
        if self.gpu_info.get('catboost_gpu', False):
            config = {
                'task_type': 'GPU',
                'devices': '0'
            }
            logger.info("Usando configuración GPU para CatBoost")
        elif fallback_to_cpu:
            config = {
                'task_type': 'CPU',
                'thread_count': -1
            }
            logger.info("Usando configuración CPU para CatBoost")
        else:
            raise RuntimeError("GPU no disponible para CatBoost y fallback deshabilitado")
        
        return config
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Obtener información de memoria disponible"""
        memory_info = {}
        
        # Memoria del sistema
        try:
            import psutil
            memory_info['system_memory'] = {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent
            }
        except ImportError:
            logger.warning("psutil no disponible para información de memoria del sistema")
        
        # Memoria GPU
        if self.cuda_available:
            try:
                import torch
                memory_info['gpu_memory'] = {
                    'total': torch.cuda.get_device_properties(0).total_memory,
                    'allocated': torch.cuda.memory_allocated(0),
                    'cached': torch.cuda.memory_reserved(0)
                }
            except Exception as e:
                logger.warning(f"Error obteniendo memoria GPU: {e}")
        
        return memory_info
    
    def print_hardware_summary(self):
        """Imprimir resumen del hardware detectado"""
        print("\n🔧======================================================================")
        print("🔧 RESUMEN DE HARDWARE DETECTADO")
        print("🔧======================================================================")
        
        print(f"🖥️  CUDA disponible: {'✅' if self.cuda_available else '❌'}")
        
        if self.cuda_available and self.gpu_info:
            print(f"🎮 GPU: {self.gpu_info.get('device_name', 'Desconocida')}")
            total_memory = self.gpu_info.get('memory_total', 0)
            if total_memory > 0:
                print(f"💾 Memoria GPU: {total_memory / 1e9:.1f} GB")
        
        print(f"\n📚 SOPORTE DE LIBRERÍAS:")
        print(f"   XGBoost GPU: {'✅' if self.gpu_info.get('xgboost_gpu', False) else '❌'}")
        print(f"   LightGBM GPU: {'✅' if self.gpu_info.get('lightgbm_gpu', False) else '❌'}")
        print(f"   CatBoost GPU: {'✅' if self.gpu_info.get('catboost_gpu', False) else '❌'}")
        
        # Información de memoria
        memory_info = self.get_memory_info()
        if 'system_memory' in memory_info:
            sys_mem = memory_info['system_memory']
            print(f"\n💾 MEMORIA SISTEMA:")
            print(f"   Total: {sys_mem['total'] / 1e9:.1f} GB")
            print(f"   Disponible: {sys_mem['available'] / 1e9:.1f} GB")
            print(f"   Uso: {sys_mem['percent']:.1f}%")

# Instancia global del GPU Manager
GPU_MANAGER = GPUManager()

import time
import psutil
import platform
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedEnergyTracker:
    """Track energy consumption and environmental impact of AI operations"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_cpu = None
        self.end_cpu = None
        self.start_memory = None
        self.end_memory = None
        
        # Energy constants (approximate values)
        self.CPU_POWER_WATTS = 15.0  # Average CPU power consumption
        self.GPU_POWER_WATTS = 50.0  # Average GPU power consumption (if available)
        self.CO2_PER_KWH = 404  # Global average grams CO2 per kWh
        
    def start_tracking(self):
        """Start tracking energy consumption"""
        try:
            self.start_time = time.time()
            self.start_cpu = psutil.cpu_percent(interval=None)
            self.start_memory = psutil.virtual_memory().used / (1024**3)  # GB
        except Exception as e:
            logger.warning(f"Could not start energy tracking: {e}")
            self.start_time = time.time()
            self.start_cpu = 0
            self.start_memory = 0
    
    def stop_tracking(self) -> Dict[str, Any]:
        """Stop tracking and calculate energy metrics"""
        try:
            self.end_time = time.time()
            self.end_cpu = psutil.cpu_percent(interval=None)
            self.end_memory = psutil.virtual_memory().used / (1024**3)  # GB
            
            return self._calculate_energy_metrics()
            
        except Exception as e:
            logger.warning(f"Could not complete energy tracking: {e}")
            return self._fallback_metrics()
    
    def _calculate_energy_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive energy metrics"""
        
        # Basic timing
        processing_time = self.end_time - self.start_time
        
        # CPU utilization change
        cpu_utilization_avg = (self.start_cpu + self.end_cpu) / 2
        
        # Memory usage
        memory_used = abs(self.end_memory - self.start_memory)
        
        # Energy calculations (simplified model)
        cpu_energy_kwh = (cpu_utilization_avg / 100) * (self.CPU_POWER_WATTS / 1000) * (processing_time / 3600)
        
        # CO2 emissions
        co2_emissions_kg = cpu_energy_kwh * (self.CO2_PER_KWH / 1000)
        co2_emissions_g = co2_emissions_kg * 1000
        
        # Energy efficiency score (0-1, higher is better)
        base_efficiency = 0.5
        if processing_time < 1.0:
            base_efficiency += 0.3
        if cpu_utilization_avg < 50:
            base_efficiency += 0.2
        
        energy_efficiency_score = min(1.0, base_efficiency)
        
        # GPU metrics (if available)
        gpu_metrics = self._get_gpu_metrics()
        
        return {
            "processing_time_seconds": processing_time,
            "cpu_utilization_start": self.start_cpu,
            "cpu_utilization_end": self.end_cpu,
            "cpu_utilization_avg": cpu_utilization_avg,
            "memory_used_gb": memory_used,
            "energy_consumed_kwh": cpu_energy_kwh,
            "co2_emissions_kg": co2_emissions_kg,
            "co2_emissions_g": co2_emissions_g,
            "energy_efficiency_score": energy_efficiency_score,
            "gpu_metrics": gpu_metrics,
            "system_info": {
                "platform": platform.system(),
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3)
            }
        }
    
    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU metrics if available"""
        try:
            # Try to get GPU info (this would require nvidia-ml-py or similar)
            # For now, return placeholder
            return {
                "gpu_available": False,
                "gpu_utilization": 0,
                "gpu_memory_used": 0,
                "gpu_power_draw": 0
            }
        except:
            return {"gpu_available": False}
    
    def _fallback_metrics(self) -> Dict[str, Any]:
        """Fallback metrics when tracking fails"""
        processing_time = (self.end_time - self.start_time) if self.end_time and self.start_time else 0.1
        
        return {
            "processing_time_seconds": processing_time,
            "cpu_utilization_start": 5.0,
            "cpu_utilization_end": 5.0,
            "cpu_utilization_avg": 5.0,
            "memory_used_gb": 0.1,
            "energy_consumed_kwh": 0.000001,
            "co2_emissions_kg": 0.0000004,
            "co2_emissions_g": 0.0004,
            "energy_efficiency_score": 0.8,
            "gpu_metrics": {"gpu_available": False},
            "system_info": {
                "platform": platform.system(),
                "cpu_count": 1,
                "total_memory_gb": 4.0
            }
        }

class EnergyBudgetManager:
    """Manage energy budgets and optimization strategies"""
    
    def __init__(self, daily_co2_budget_g: float = 100.0):
        self.daily_co2_budget_g = daily_co2_budget_g
        self.current_usage_g = 0.0
        self.reset_time = time.time()
    
    def check_budget(self, estimated_co2_g: float) -> Dict[str, Any]:
        """Check if operation fits within energy budget"""
        
        # Reset daily usage if needed
        if time.time() - self.reset_time > 86400:  # 24 hours
            self.current_usage_g = 0.0
            self.reset_time = time.time()
        
        remaining_budget = self.daily_co2_budget_g - self.current_usage_g
        can_proceed = estimated_co2_g <= remaining_budget
        
        return {
            "can_proceed": can_proceed,
            "estimated_co2_g": estimated_co2_g,
            "current_usage_g": self.current_usage_g,
            "remaining_budget_g": remaining_budget,
            "budget_utilization_percent": (self.current_usage_g / self.daily_co2_budget_g) * 100,
            "recommendation": self._get_recommendation(estimated_co2_g, remaining_budget)
        }
    
    def update_usage(self, actual_co2_g: float):
        """Update actual usage"""
        self.current_usage_g += actual_co2_g
    
    def _get_recommendation(self, estimated_co2_g: float, remaining_budget: float) -> str:
        """Get energy optimization recommendation"""
        
        if estimated_co2_g > remaining_budget:
            return "❌ Exceeds daily budget - consider using lighter model"
        elif estimated_co2_g > remaining_budget * 0.8:
            return "⚠️ High energy usage - monitor remaining budget"
        elif estimated_co2_g < remaining_budget * 0.1:
            return "✅ Low energy usage - efficient operation"
        else:
            return "✅ Within acceptable energy range"

# Singleton instances
energy_tracker = EnhancedEnergyTracker()
energy_budget_manager = EnergyBudgetManager()




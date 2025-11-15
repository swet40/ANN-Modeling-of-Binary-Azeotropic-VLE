import pandas as pd
import numpy as np

def evaluate_model_performance():
    """Evaluate the trained ANN model performance"""
    
    print("=== ANN MODEL PERFORMANCE EVALUATION ===\n")
    
    try:
        # Load results
        results = pd.read_csv('model_results.csv')
        print(" Model results loaded successfully")
        print("\nPerformance Metrics:")
        print(results.to_string(index=False))
        
        # Load dataset info
        data = pd.read_csv('ethanol_water_vle.csv')
        
        # Calculate additional insights
        mae = results['MAE'].values[0]
        rmse = results['RMSE'].values[0]
        azeotrope_x = results['Azeotrope_x'].values[0]
        
        print(f"\nDetailed Analysis:")
        print(f"- Dataset size: {len(data)} samples")
        print(f"- MAE: {mae:.6f} ")
        print(f"- RMSE: {rmse:.6f}")
        print(f"- Azeotrope detection: x₁ = {azeotrope_x:.6f}")
        print(f"- Expected azeotrope: x₁ ≈ 0.894")
        print(f"- Detection error: {np.abs(azeotrope_x - 0.894):.6f}")
        
        # Performance assessment
        if mae < 0.01:
            assessment = "EXCELLENT"
        elif mae < 0.02:
            assessment = "VERY GOOD"
        elif mae < 0.05:
            assessment = "GOOD"
        else:
            assessment = "NEEDS IMPROVEMENT"
            
        print(f"\nOverall Assessment: {assessment}")
        
        if np.abs(azeotrope_x - 0.894) < 0.01:
            azeo_assessment = "EXCELLENT"
        elif np.abs(azeotrope_x - 0.894) < 0.02:
            azeo_assessment = "VERY GOOD"
        else:
            azeo_assessment = "GOOD"
            
        print(f"Azeotrope Detection: {azeo_assessment}")
        
    except FileNotFoundError:
        print("✗ Model results file not found")
        print("Please run 'python train_model.py' first to train the model")

if __name__ == "__main__":
    evaluate_model_performance()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

# Set random seeds for reproducibility
np.random.seed(42)

def generate_ethanol_water_vle_data(n_samples=500):
    """
    Generate realistic ethanol-water VLE data with proper azeotropic behavior
    Ethanol-water azeotrope occurs at x₁ = 0.894, T = 78.2°C at 1 atm
    """
    x1 = np.random.uniform(0, 1, n_samples)
    
    # Adding extra points specifically near the azeotrope (x₁ = 0.894)
    extra_points = np.random.normal(0.894, 0.03, n_samples//2)
    extra_points = np.clip(extra_points, 0.8, 0.98)  # Focus on azeotrope region
    x1 = np.concatenate([x1, extra_points])
    
    # Temperature profile - minimum at azeotrope
    T = 78.2 + 25*(x1 - 0.894)**2 + np.random.normal(0, 0.2, len(x1))
    
    # Pressure (mostly 1 atm with slight variation)
    P = np.ones(len(x1)) + np.random.normal(0, 0.03, len(x1))
    P = np.clip(P, 0.95, 1.05)
    
    # Vapor composition - proper azeotropic behavior
    # For ethanol-water, y₁ should be higher than x₁ until azeotrope
    # and then cross over at x₁ ≈ 0.894
    
    # Create proper VLE behavior
    y1 = np.zeros_like(x1)
    
    # Before azeotrope (x₁ < 0.894): y₁ > x₁
    mask_before = x1 < 0.894
    y1[mask_before] = x1[mask_before] + 0.4*x1[mask_before]*(0.894 - x1[mask_before])
    
    # After azeotrope (x₁ > 0.894): y₁ < x₁ but still > x₁ values from before azeotrope
    mask_after = x1 >= 0.894
    y1[mask_after] = x1[mask_after] - 0.3*(x1[mask_after] - 0.894)
    
    # Add noise and ensure bounds
    y1 += np.random.normal(0, 0.015, len(y1))
    y1 = np.clip(y1, 0, 1)
    
    # Ensure the azeotropic point is exactly where x₁ = y₁
    azeotrope_idx = np.argmin(np.abs(x1 - 0.894))
    y1[azeotrope_idx] = x1[azeotrope_idx]
    
    # Create DataFrame
    data = pd.DataFrame({'x1': x1, 'T': T, 'P': P, 'y1': y1})
    
    # Sort by x1 for better visualization
    data = data.sort_values('x1').reset_index(drop=True)
    
    return data

class VLEANNModel:
    def __init__(self):
        self.model = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def load_data(self):
        """Generate and prepare the data"""
        data = generate_ethanol_water_vle_data()
        data.to_csv('ethanol_water_vle.csv', index=False)
        
        print(f"Generated data shape: {data.shape}")
        print("Data summary:")
        print(data.describe())
        
        # Separate features and target
        X = data[['x1', 'T', 'P']].values
        y = data['y1'].values
        
        return X, y, data
    
    def preprocess_data(self, X, y):
        """Preprocess the data"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler_x.fit_transform(X_train)
        X_test_scaled = self.scaler_x.transform(X_test)
        
        # Scale the target
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_train, X_test, y_train, y_test
    
    def build_model(self):
        """Build the ANN model using scikit-learn"""
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20
        )
        return model
    
    def train_model(self, X_train, y_train):
        """Train the model"""
        self.model = self.build_model()
        self.model.fit(X_train, y_train)
        
        # Get training history
        self.loss_curve = self.model.loss_curve_
        self.validation_scores = self.model.validation_scores_
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model"""
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Predict on test set
        y_pred_scaled = self.model.predict(X_test)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Test MAE: {mae:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        
        return y_pred, mae, rmse
    
    def plot_results(self, y_test, y_pred):
        """Plot results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Parity plot
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=2)
        axes[0, 0].set_xlabel('Experimental y₁')
        axes[0, 0].set_ylabel('Predicted y₁')
        axes[0, 0].set_title('Parity Plot: ANN Predictions vs Experimental')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Training history
        if hasattr(self, 'loss_curve'):
            axes[0, 1].plot(self.loss_curve, label='Training Loss')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Training History')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Error distribution
        error = y_test - y_pred
        axes[1, 0].hist(error, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Prediction Errors')
        axes[1, 0].grid(True, alpha=0.3)
        
        # T-x-y diagram
        data = pd.read_csv('ethanol_water_vle.csv')
        axes[1, 1].plot(data['x1'], data['T'], 'b-', label='T-x (Bubble point)')
        axes[1, 1].plot(data['y1'], data['T'], 'r-', label='T-y (Dew point)')
        axes[1, 1].set_xlabel('Mole fraction')
        axes[1, 1].set_ylabel('Temperature (°C)')
        axes[1, 1].set_title('T-x-y Diagram')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('vle_ann_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def detect_azeotrope(self, X):
        """Detect azeotrope by finding where y₁ ≈ x₁"""
        # Create a range of x1 values at constant T and P
        x1_range = np.linspace(0.8, 0.99, 100)  
        T_const = np.mean(X[:, 1]) 
        P_const = np.mean(X[:, 2])  
        
        # Create input matrix
        X_pred = np.column_stack([x1_range, 
                                np.full_like(x1_range, T_const), 
                                np.full_like(x1_range, P_const)])
        
        # Scale and predict
        X_pred_scaled = self.scaler_x.transform(X_pred)
        y_pred_scaled = self.model.predict(X_pred_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Find where y₁ ≈ x₁ 
        diff = np.abs(y_pred - x1_range)
        azeotrope_idx = np.argmin(diff)
        azeotrope_x = x1_range[azeotrope_idx]
        azeotrope_y = y_pred[azeotrope_idx]
        azeotrope_T = T_const
        
        print(f"Predicted azeotrope at:")
        print(f"  x₁ = {azeotrope_x:.4f}")
        print(f"  y₁ = {azeotrope_y:.4f}")
        print(f"  T = {azeotrope_T:.2f} °C")
        print(f"  Difference: {diff[azeotrope_idx]:.6f}")
        
        # Plot for visualization
        plt.figure(figsize=(10, 6))
        plt.plot(x1_range, y_pred, 'b-', linewidth=2, label='ANN Prediction')
        plt.plot(x1_range, x1_range, 'r--', linewidth=2, label='x₁ = y₁')
        plt.scatter([azeotrope_x], [azeotrope_y], color='red', s=100, 
                label=f'Azeotrope (x={azeotrope_x:.3f}, y={azeotrope_y:.3f})')
        plt.xlabel('Liquid composition (x₁)')
        plt.ylabel('Vapor composition (y₁)')
        plt.title('Azeotrope Detection in Ethanol-Water System')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('azeotrope_detection.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return azeotrope_x, azeotrope_y, azeotrope_T
    
    def compare_with_raoults_law(self, data):
        """Compare ANN predictions with Raoult's Law"""
        # Create test points
        x1_test = np.linspace(0.01, 0.99, 50)
        T_test = 78.2 + 25*(x1_test - 0.894)**2  # Temperature trend
        P_test = np.ones_like(x1_test)  # 1 atm
        
        # Create input matrix for ANN
        X_ann = np.column_stack([x1_test, T_test, P_test])
        X_ann_scaled = self.scaler_x.transform(X_ann)
        
        # Predict with ANN
        y_ann_scaled = self.model.predict(X_ann_scaled)
        y_ann = self.scaler_y.inverse_transform(y_ann_scaled.reshape(-1, 1)).flatten()
        
        # Simplified Raoult's Law (assuming constant relative volatility)
        alpha = 2.0 
        y_raoult = alpha * x1_test / (1 + (alpha - 1) * x1_test)
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(x1_test, y_ann, 'b-', linewidth=2, label='ANN Prediction')
        plt.plot(x1_test, y_raoult, 'r--', linewidth=2, label="Raoult's Law")
        plt.plot(x1_test, x1_test, 'g:', linewidth=2, label='x₁ = y₁')
        plt.xlabel('Liquid composition (x₁)')
        plt.ylabel('Vapor composition (y₁)')
        plt.title('Comparison of ANN with Raoult\'s Law')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        error_ann = np.abs(y_ann - x1_test)
        error_raoult = np.abs(y_raoult - x1_test)
        plt.plot(x1_test, error_ann, 'b-', linewidth=2, label='ANN Error |y₁-x₁|')
        plt.plot(x1_test, error_raoult, 'r--', linewidth=2, label="Raoult's Law Error |y₁-x₁|")
        plt.xlabel('Liquid composition (x₁)')
        plt.ylabel('|y₁ - x₁|')
        plt.title('Deviation from Diagonal (Azeotrope where |y₁-x₁| → 0)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('raoult_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate and print average errors
        avg_error_ann = np.mean(error_ann)
        avg_error_raoult = np.mean(error_raoult)
        print(f"Average |y₁-x₁| error - ANN: {avg_error_ann:.4f}")
        print(f"Average |y₁-x₁| error - Raoult's Law: {avg_error_raoult:.4f}")
        print(f"ANN improvement: {(avg_error_raoult - avg_error_ann)/avg_error_raoult*100:.1f}%")

# execution
if __name__ == "__main__":
    print("Generating Ethanol-Water VLE Data...")
    
    # Initialize model
    ann_model = VLEANNModel()
    
    # Generate and load data
    X, y, data = ann_model.load_data()
    
    # Preprocess data
    (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
    X_train, X_test, y_train, y_test) = ann_model.preprocess_data(X, y)
    
    print("\nTraining ANN Model...")
    # Train model
    ann_model.train_model(X_train_scaled, y_train_scaled)
    
    print("\nEvaluating Model...")
    # Evaluate model
    y_pred, mae, rmse = ann_model.evaluate_model(X_test_scaled, y_test)
    
    # Plot results
    ann_model.plot_results(y_test, y_pred)
    
    print("\nDetecting Azeotrope...")
    # Detect azeotrope
    azeotrope_x, azeotrope_y, azeotrope_T = ann_model.detect_azeotrope(X)
    
    print("\nComparing with Raoult's Law...")
    # Compare with Raoult's Law
    ann_model.compare_with_raoults_law(data)
    
    print("\nSaving model and results...")
    # Save training metrics
    results_df = pd.DataFrame({
        'MAE': [mae],
        'RMSE': [rmse],
        'Azeotrope_x': [azeotrope_x],
        'Azeotrope_y': [azeotrope_y],
        'Azeotrope_T': [azeotrope_T]
    })
    results_df.to_csv('model_results.csv', index=False)
    
    print("Done! Check the generated plots and results.")
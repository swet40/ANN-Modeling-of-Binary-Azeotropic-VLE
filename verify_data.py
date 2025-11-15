import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def verify_dataset():
    """Comprehensive verification of the generated VLE dataset"""
    
    print("=== ETHANOL-WATER VLE DATASET VERIFICATION ===\n")
    
    # Load the dataset
    try:
        data = pd.read_csv('ethanol_water_vle.csv')
        print(f"✓ Dataset loaded successfully: {data.shape[0]} samples, {data.shape[1]} features")
    except FileNotFoundError:
        print("✗ Dataset file not found. Please run the main script first.")
        return
    
    # Check basic structure
    print("\n1. DATASET STRUCTURE:")
    required_columns = ['x1', 'T', 'P', 'y1']
    if all(col in data.columns for col in required_columns):
        print("✓ All required columns present")
    else:
        print("✗ Missing required columns")
        return
    
    # Check data ranges and validity
    print("\n2. DATA VALIDITY CHECK:")
    
    # Check composition ranges (should be 0-1)
    x1_valid = data['x1'].between(0, 1).all()
    y1_valid = data['y1'].between(0, 1).all()
    print(f"✓ x1 values in [0,1] range: {x1_valid}")
    print(f"✓ y1 values in [0,1] range: {y1_valid}")
    
    # Check temperature (should be reasonable for ethanol-water)
    T_valid = data['T'].between(70, 100).all()
    print(f"✓ Temperature values in reasonable range [70-100°C]: {T_valid}")
    
    # Check pressure (should be around 1 atm)
    P_valid = data['P'].between(0.9, 1.1).all()
    print(f"✓ Pressure values around 1 atm: {P_valid}")
    
    # Statistical summary
    print("\n3. STATISTICAL SUMMARY:")
    print(data.describe())
    
    # Check for azeotropic behavior
    print("\n4. AZEOTROPE ANALYSIS:")
    
    # Find points where x1 ≈ y1 (azeotrope condition)
    tolerance = 0.05  # 5% tolerance
    azeotrope_candidates = data[np.abs(data['x1'] - data['y1']) < tolerance]
    
    if len(azeotrope_candidates) > 0:
        print(f"✓ Found {len(azeotrope_candidates)} potential azeotrope points")
        
        # Find the point closest to x1 = y1
        idx_min = np.argmin(np.abs(azeotrope_candidates['x1'] - azeotrope_candidates['y1']))
        azeotrope_point = azeotrope_candidates.iloc[idx_min]
        
        print(f"✓ Closest azeotrope point:")
        print(f"   x1 = {azeotrope_point['x1']:.4f}")
        print(f"   y1 = {azeotrope_point['y1']:.4f}")
        print(f"   T = {azeotrope_point['T']:.2f} °C")
        print(f"   P = {azeotrope_point['P']:.3f} atm")
        print(f"   Difference |x1-y1| = {np.abs(azeotrope_point['x1'] - azeotrope_point['y1']):.6f}")
        
        # Check if it's near the expected ethanol-water azeotrope (x1 ≈ 0.894)
        expected_azeotrope = 0.894
        if np.abs(azeotrope_point['x1'] - expected_azeotrope) < 0.05:
            print("✓ Azeotrope composition is near expected value (x1 ≈ 0.894)")
        else:
            print("⚠ Azeotrope composition differs from expected value")
    else:
        print("✗ No azeotrope points found (no points where x1 ≈ y1)")
    
    # Check data distribution near azeotrope
    print("\n5. DATA DISTRIBUTION ANALYSIS:")
    
    azeotrope_region = data[(data['x1'] > 0.84) & (data['x1'] < 0.94)]
    print(f"✓ Samples in azeotrope region (0.84 < x1 < 0.94): {len(azeotrope_region)}")
    
    if len(azeotrope_region) < 10:
        print("⚠ Warning: Few samples in azeotrope region")
    else:
        print("✓ Good sampling in azeotrope region")
    
    # Generate verification plots
    print("\n6. GENERATING VERIFICATION PLOTS...")
    
    # Convert pandas Series to numpy arrays to avoid the compatibility issue
    x1_values = data['x1'].values
    y1_values = data['y1'].values
    T_values = data['T'].values
    
    # Create a comprehensive verification figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Plot 1: x-y diagram
    axes[0, 0].plot(x1_values, y1_values, 'bo', alpha=0.5, label='VLE Data')
    axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='x₁ = y₁')
    if len(azeotrope_candidates) > 0:
        axes[0, 0].plot(azeotrope_point['x1'], azeotrope_point['y1'], 'ro', 
                       markersize=8, label='Azeotrope')
    axes[0, 0].set_xlabel('Liquid composition (x₁)', fontsize=12)
    axes[0, 0].set_ylabel('Vapor composition (y₁)', fontsize=12)
    axes[0, 0].set_title('x-y Diagram', fontsize=14, pad=20)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: T-x-y diagram
    axes[0, 1].plot(x1_values, T_values, 'b-', label='T-x (Bubble point)', linewidth=2)
    axes[0, 1].plot(y1_values, T_values, 'r-', label='T-y (Dew point)', linewidth=2)
    if len(azeotrope_candidates) > 0:
        axes[0, 1].plot(azeotrope_point['x1'], azeotrope_point['T'], 'ro', 
                       markersize=8, label='Azeotrope')
    axes[0, 1].set_xlabel('Mole fraction', fontsize=12)
    axes[0, 1].set_ylabel('Temperature (°C)', fontsize=12)
    axes[0, 1].set_title('T-x-y Diagram', fontsize=14, pad=20)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Composition distribution
    axes[1, 0].hist(x1_values, bins=30, alpha=0.7, color='blue', edgecolor='black', label='x₁')
    axes[1, 0].hist(y1_values, bins=30, alpha=0.7, color='red', edgecolor='black', label='y₁')
    axes[1, 0].axvline(x=expected_azeotrope, color='green', linestyle='--', 
                      label=f'Expected azeotrope (x={expected_azeotrope})')
    axes[1, 0].set_xlabel('Composition', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Composition Distribution', fontsize=14, pad=20)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Temperature distribution
    axes[1, 1].hist(T_values, bins=30, alpha=0.7, color='green', edgecolor='black')
    if len(azeotrope_candidates) > 0:
        axes[1, 1].axvline(x=azeotrope_point['T'], color='red', linestyle='--', 
                          label=f'Azeotrope T ({azeotrope_point["T"]:.1f}°C)')
    axes[1, 1].set_xlabel('Temperature (°C)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Temperature Distribution', fontsize=14, pad=20)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.savefig('dataset_verification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(" Verification plots saved as 'dataset_verification.png'")
    
    # Final assessment
    print("\n7. FINAL ASSESSMENT:")
    issues = []
    
    if not all([x1_valid, y1_valid, T_valid, P_valid]):
        issues.append("Data range validation failed")
    
    if len(azeotrope_candidates) == 0:
        issues.append("No azeotrope points detected")
    
    if len(azeotrope_region) < 10:
        issues.append("Insufficient sampling in azeotrope region")
    
    if issues:
        print(" Dataset has some issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print(" Dataset validation passed successfully!")
        print(" All criteria met for binary azeotropic VLE data")
    
    return data, azeotrope_point if len(azeotrope_candidates) > 0 else None

if __name__ == "__main__":
    verify_dataset()
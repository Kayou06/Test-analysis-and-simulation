import CoolProp.CoolProp as CP

def compressibility_factor_and_density():
    temperature = [222, 220.6, 252, 252.2, 250.7]
    t_kelvin = [temp + 273.15 for temp in temperature]

    pressure = [8.9, 12.1, 16.1, 12.1, 9]
    P_pascal = [p * 1e5 for p in pressure]

    Z_values = []
    rho_values = []

    for T, P in zip(t_kelvin, P_pascal):
        Z = CP.PropsSI('Z', 'T', T, 'P', P, 'MM')
        rho = CP.PropsSI('D', 'T', T, 'P', P, 'MM')  # density in kg/m³

        Z_values.append(Z)
        rho_values.append(rho)

    return Z_values, rho_values


Z, rho = compressibility_factor_and_density()

print("Z:", Z)
print("rho:", rho)

# The output of the function is a list with the compressibility factor values for each corresponding temperature and pressure.
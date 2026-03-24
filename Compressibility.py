import CoolProp.CoolProp as CP

def compressibility_factor_and_density():
    temperature =  temperature = [221.8,219.8, 248.5, 248.5, 254, 252.9,251]
    t_kelvin = [temp + 273.15 for temp in temperature]

    pressure = [ 8.09, 12.04,  18.50, 19.95,15.9,12.06,9.08 ]
    P_pascal = [p * 1e5 for p in pressure]

    Z_values = []
    rho0_values = []

    for T, P in zip(t_kelvin, P_pascal):
        Z = CP.PropsSI('Z', 'T', T, 'P', P, 'MM')
        rho = CP.PropsSI('D', 'T', T, 'P', P, 'MM')  # actual density in kg/m³
        

        Z_values.append(Z)
        rho0_values.append(rho)
        

    return Z_values, rho0_values


Z, rho0 = compressibility_factor_and_density()

print("Z:", Z)
print("rho0:", rho0)
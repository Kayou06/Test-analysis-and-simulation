import CoolProp.CoolProp as CP

def compressibility_factor():
  
    temperature = [222, 220.6, 252, 252.2, 250.7]
    t_kelvin = [temp + 273.15 for temp in temperature]
    #print (t_kelvin)
    pressure = [8.9, 12.1, 16.1, 12.1, 9]
    P_pascal = [p * 1e5 for p in pressure]
    Z_values = [CP.PropsSI('Z', 'T', T, 'P', P, 'MM') for T, P in zip(t_kelvin, P_pascal)]
    return Z_values
print(compressibility_factor())

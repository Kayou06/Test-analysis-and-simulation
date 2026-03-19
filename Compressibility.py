import CoolProp.CoolProp as CP

def compressibility_factor():
  
    temperature = [221.8,219.8, 248.5, 248.5, 254, 252.9,251]
    t_kelvin = [temp + 273.15 for temp in temperature]
    #print (t_kelvin)
    pressure = [ 8.09, 12.04,  18.50, 19.95,15.9,12.06,9.08 ]
    P_pascal = [p * 1e5 for p in pressure]
    Z_values = [CP.PropsSI('Z', 'T', T, 'P', P, 'MM') for T, P in zip(t_kelvin, P_pascal)]
    return Z_values
print(compressibility_factor())
# %%
## Se importan las librerias necesarias
import math
import numpy as np
import random
from deap import base
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
import time

# %%
## Valores Comerciales
# Se generan dos listas con los valores comerciales de Capacitores y Resistores
ResList = [1.0, 10, 100, 1.0e3, 10e3, 100e3, 1.0e6,
           1.1, 11, 110, 1.1e3, 11e3, 110e3, 1.1e6,
           1.2, 12, 120, 1.2e3, 12e3, 120e3, 1.2e6,
           1.3, 13, 130, 1.3e3, 13e3, 130e3, 1.3e6,
           1.5, 15, 150, 1.5e3, 15e3, 150e3, 1.5e6,
           1.6, 16, 160, 1.6e3, 16e3, 160e3, 1.6e6,
           1.8, 18, 180, 1.8e3, 18e3, 180e3, 1.8e6,
           2.0, 20, 200, 2.0e3, 20e3, 200e3, 2.0e6,
           2.2, 22, 220, 2.2e3, 22e3, 220e3, 2.2e6,
           2.4, 24, 240, 2.4e3, 24e3, 240e3, 2.4e6,
           2.7, 27, 270, 2.7e3, 27e3, 270e3, 2.7e6,
           3.0, 30, 300, 3.0e3, 30e3, 300e3, 3.0e6,
           3.3, 33, 330, 3.3e3, 33e3, 330e3, 3.3e6,
           3.6, 36, 360, 3.6e3, 36e3, 360e3, 3.6e6,
           3.9, 39, 390, 3.9e3, 39e3, 390e3, 3.9e6,
           4.3, 43, 430, 4.3e3, 43e3, 430e3, 4.3e6,
           4.7, 47, 470, 4.7e3, 47e3, 470e3, 4.7e6,
           5.1, 51, 510, 5.1e3, 51e3, 510e3, 5.1e6,
           5.6, 56, 560, 5.6e3, 56e3, 560e3, 5.6e6,
           6.2, 62, 620, 6.2e3, 62e3, 620e3, 6.2e6,
           6.8, 68, 680, 6.8e3, 68e3, 680e3, 6.8e6,
           7.5, 75, 750, 7.5e3, 75e3, 750e3, 7.5e6,
           8.2, 82, 820, 8.2e3, 82e3, 820e3, 8.2e6,
           9.1, 91, 910, 9.1e3, 91e3, 910e3, 9.1e6]
CapList = [10e-12, 100e-12, 1000e-12, 0.010e-6, 0.10e-6, 1.0e-6, 10e-6,
           12e-12, 120e-12, 1200e-12, 0.012e-6, 0.12e-6, 1.2e-6,
           15e-12, 150e-12, 1500e-12, 0.015e-6, 0.15e-6, 1.5e-6,
           18e-12, 180e-12, 1800e-12, 0.018e-6, 0.18e-6, 1.8e-6,
           22e-12, 220e-12, 2200e-12, 0.022e-6, 0.22e-6, 2.2e-6, 22e-6,
           27e-12, 270e-12, 2700e-12, 0.027e-6, 0.27e-6, 2.7e-6,
           33e-12, 330e-12, 3300e-12, 0.033e-6, 0.33e-6, 3.3e-6, 33e-6,
           39e-12, 390e-12, 3900e-12, 0.039e-6, 0.39e-6, 3.9e-6,
           47e-12, 470e-12, 4700e-12, 0.047e-6, 0.47e-6, 4.7e-6, 47e-6,
           56e-12, 560e-12, 5600e-12, 0.056e-6, 0.56e-6, 5.6e-6,
           68e-12, 680e-12, 6800e-12, 0.068e-6, 0.68e-6, 6.8e-6,
           82e-12, 820e-12, 8200e-12, 0.082e-6, 0.82e-6, 8.2e-6]

# %%
## Creacion de funciones importantes 

# Cromosoma de la forma [R1,R2,RF,C1,C2, RA, RB]
# Creacion de individuos utilizando solo valores comerciales
def chromocreate():
    Res = random.choices(ResList,k=3)
    Cap = random.choices(CapList,k=2)
    ResAB = random.choices(ResList,k=2)
    return Res + Cap + ResAB

# Funcion para calcular la frecuencia central
def centralfreq(individual):
    den = individual[0][0]*individual[0][1]*individual[0][2]*individual[0][3]*individual[0][4]
    num = individual[0][0]+individual[0][2]
    cons = 1/(2*math.pi)
    fo = cons*math.sqrt(num/den)
    return fo
    
# Funcion de Calidad
def evalFct(individual):
    fo = centralfreq(individual)   
    # Calidad basada en minimizar el error (pero siendo una funcion para maximizar)
    fc = -abs(10000-fo)
    #print(fc)
    return fc,

# Funcion para calcular Q
def quality_func(individual):
    R1 = individual[0][0]
    R2 = individual[0][1]
    RF = individual[0][2]
    C1 = individual[0][3]
    C2 = individual[0][4]
    RA = individual[0][5]
    RB = individual[0][6]
    num_Q = (R1+RF)*R1*RF*R2*C1*C2
    den_Q = R1*RF*(C1 + C2) + R2*C2*(RF - (RB/RA)*R1)
    Q = np.sqrt(num_Q)/den_Q
    return Q

# Funcion para evaluar individuos si se tienen que penalizar 
def feasible(individual):
    Q = quality_func(individual)
    if Q > 2.5:
        return True
    return False
    
# Funcion customizada de mutación para mantener parametros necesarios
def custommutate(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            if i <= 2:
                individual[i] = random.choice(ResList)
            else:
                individual[i] = random.choice(CapList)

    return individual,

# %%
## Diseñar DEAP toolbox
# Se crea como problema de maximizacion
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Se registra funcion de creacion de individuos
toolbox.register("chromocreate", chromocreate)

toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.chromocreate,n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Caracteristicas del evolutivo
toolbox.register("evaluate", evalFct)
toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, -10000.0))

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custommutate, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# %%
## Ejemplo de una poblacion de 10 individuos
print(toolbox.population(n=10))

# %%
## Log de datos
log = tools.Logbook()


# %%
## Algoritmo Evolutivo 
def main():
    # Se crea una poblacion de tamaño 'n'
    pop = toolbox.population(n=50000)
    
    # Se evalua la poblacion
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # CXPB: probabilidad de cruce
    # MUTPB: probabilidad de mutación
    CXPB, MUTPB = 0.5, 0.1
    
    # Se crea una lista con todos los valores de fitness 
    fits = [ind.fitness.values[0] for ind in pop]
    
    # Variable de generaciones
    g = 0
    
    # Comienza la evolucion, tambien se pueden configurar criterios de paro
    while g < 100:
        # Nueva generacion
        g = g + 1
        #print("-- Generation %i --" % g)
        
        # Se hace la selección
        offspring = toolbox.select(pop, len(pop))
        # Se clonan los individuos seleccionados
        offspring = list(map(toolbox.clone, offspring))
        
        # Se aplica la mutación y el cruce
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1[0], child2[0])
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant[0])
                del mutant.fitness.values
            
        # Los nuevos indviduos se marcan como fitness invalidos y se vuelven a evaluar
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Se remplaza toda la población con los nuevos hijos
        pop[:] = offspring
        
        # Calculo de diferentes valores y calculo de desviacion estandar
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        best_in_gen = tools.selBest(pop, 1)[0]
        
        # Se almacena el valor de generacion, mejor individuo, fitness y desviacion estandar en el log.
        log.record(generation=g, ind=best_in_gen, fit = best_in_gen.fitness.getValues()[0], std_dev=std)
    
    # El retorno de la funcion es el mejor individuo de la ultima generacion
    best = tools.selBest(pop, 1)[0]
    return best

# %%
## Se corre el algoritmo
start = time.time()
best_solution = main()
end = time.time()
print("Time elapsed: ", round((end - start),2), "s")


# %%
## Grafico de valores importantes

gen = log.select("generation")
fit_max = log.select("fit")
std_dev = log.select("std_dev")

fig, ax1 = plt.subplots()
line1 = ax1.plot(gen, fit_max, "b-", label="Maximum Fitness")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness", color="b")
#ax1.ticklabel_format(style='plain')
for tl in ax1.get_yticklabels():
    tl.set_color("b")

ax2 = ax1.twinx()
line2 = ax2.plot(gen, std_dev, "r-", label="Standard Deviation")
ax2.set_ylabel("Desviación", color="r")
#ax2.ticklabel_format(style='plain')
for tl in ax2.get_yticklabels():
    tl.set_color("r")

lns = line1 + line2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="center right")

plt.show()

# %%
## Los resultados finales:

print ("Cromosoma de forma: [R1,R2,RF,C1,C2, RA, RB]")
print ("Mejor individuo:   ", log.select("ind")[99])
print ("Con un fitness de: ", log.select("fit")[99])

# %%
## Calculo de valores importantes al problema
print("La frecuencia central es: ", round(centralfreq(best_solution),3))
print("La calidad 'Q' es: ", round(quality_func(best_solution),2))


# %%
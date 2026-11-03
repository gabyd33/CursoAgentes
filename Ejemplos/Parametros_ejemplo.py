from ollama import chat

for temp in [0.0, 0.7, 1.5]:
    print("Current temperature:", temp)
    resp = chat(
        model="qwen3:8b",
        messages=[
            {
                "role": "user",
                "content": "Listá 3 ventajas de usar RAG en vez de fine-tuning, una por línea."
            }
        ],
        options={
            "temperature": temp,
            "top_p": 0.9,
            "num_predict": 900#,      # equivalente aproximado a maxTokens
            #"stop": ["\n\n"]         # equivalente a stopSequences
        }
    )


    print(resp["message"]["content"])
    print("\nStop reason:", resp.get("done_reason"))


#Current temperature: 0.0
#1. **Acceso a conocimiento actualizado sin reentrenamiento**: RAG permite integrar información externa en tiempo real mediante la recuperación de documentos, evitando la necesidad de reentrenar el modelo para mantenerlo al día con nuevos datos.  
#2. **Eficiencia en recursos**: RAG utiliza modelos preentrenados y solo requiere ajustes en la etapa de recuperación, reduciendo la carga computacional comparada con el fine-tuning, que implica entrenar desde cero.  
#3. **Adaptabilidad a dominios específicos**: RAG puede especializarse en tareas con conocimientos técnicos o dominios específicos al recuperar documentos relevantes, sin necesidad de grandes conjuntos de datos etiquetados para el fine-tuning.

#Stop reason: stop
#Current temperature: 0.7
#1. **Acceso a información actualizada sin reentrenamiento**: RAG permite integrar conocimientos recientes o específicos al momento de generarrespuestas, mientras que el fine-tuning requiere reentrenar el modelo con nuevos datos, lo que puede ser lento y costoso.  
#2. **Flexibilidad en la gestión de dominios**: RAG puede adaptarse a distintos temas o contextos mediante la recuperación de documentos relevantes, mientras que el fine-tuning suele requerir datos específicos para cada tarea.  
#3. **Reducción de la dependencia de datos etiquetados**: RAG utiliza la recuperación de información para complementar la generación, minimizando la necesidad de grandes conjuntos de datos etiquetados, que son críticos en el fine-tuning.

#Stop reason: stop
#Current temperature: 1.5
#1. **Acceso a información actualizada**: RAG puede incorporar datos recientes de fuentes externas, mientras que el fine-tuning depende de losdatos de entrenamiento, que pueden estar desactualizados.  
#2. **Reducción de costos y esfuerzo de entrenamiento**: RAG no requiere reentrenar el modelo completo, solo actualizar la base de datos de recuperación, mientras que el fine-tuning implica recursos computacionales intensivos.  
#3. **Flexibilidad en la adaptación**: RAG permite personalizar la respuesta según la necesidad del usuario mediante la selección de fuentes, mientras que el fine-tuning ajusta la capacidad del modelo de forma fija.
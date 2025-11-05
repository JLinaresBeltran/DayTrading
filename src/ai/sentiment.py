"""
Módulo para análisis de sentimiento de noticias usando transformers.
"""


def load_sentiment_analyzer(model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
    """
    Carga el pipeline de análisis de sentimiento.

    Args:
        model_name: Nombre del modelo de Hugging Face

    Returns:
        Pipeline de sentiment analysis
    """
    try:
        from transformers import pipeline
        analyzer = pipeline("sentiment-analysis", model=model_name)
        print(f"✓ Modelo de sentimiento cargado: {model_name}")
        return analyzer
    except Exception as e:
        print(f"⚠ No se pudo cargar modelo de sentimiento: {e}")
        return None


def get_sentiment_score(news_headlines=None, analyzer=None):
    """
    Analiza el sentimiento de titulares de noticias.

    Args:
        news_headlines: Lista de titulares (si None, usa titulares simulados)
        analyzer: Pipeline de sentiment analysis

    Returns:
        Puntaje promedio de sentimiento (1.0 a 5.0)
    """
    if analyzer is None:
        return 3.0  # Neutral si no hay analizador

    # Si no hay noticias, usar simuladas (en producción, usar API de noticias)
    if news_headlines is None or len(news_headlines) == 0:
        news_headlines = [
            "Bitcoin reaches new milestone in crypto market",
            "Investors show confidence in digital assets",
            "Cryptocurrency market shows steady growth",
            "Blockchain technology adoption increases",
            "Market analysts predict positive trends"
        ]

    try:
        # Analizar sentimiento
        results = analyzer(news_headlines)

        # Convertir labels a puntajes numéricos (1-5 estrellas)
        scores = []
        for result in results:
            label = result['label']
            # Labels típicos: "1 star", "2 stars", etc.
            score = int(label.split()[0])
            scores.append(score)

        # Promedio
        avg_score = sum(scores) / len(scores) if scores else 3.0

        return round(avg_score, 2)

    except Exception as e:
        print(f"Error en análisis de sentimiento: {e}")
        return 3.0  # Neutral en caso de error


if __name__ == "__main__":
    print("=== Test de Sentiment Analysis ===\n")

    # Cargar analyzer
    print("1. Cargando modelo de sentimiento...")
    analyzer = load_sentiment_analyzer()

    if analyzer:
        # Test con noticias de ejemplo
        print("\n2. Analizando noticias de ejemplo...")
        news = [
            "Bitcoin price surges to new all-time high",
            "Crypto market faces regulatory concerns",
            "Institutional investors embrace digital assets",
            "Market volatility increases amid uncertainty",
            "Blockchain adoption continues to grow"
        ]

        score = get_sentiment_score(news, analyzer)
        print(f"\n   Puntaje de sentimiento: {score}/5.0")

        if score > 3.5:
            print("   Sentimiento: POSITIVO")
        elif score < 2.5:
            print("   Sentimiento: NEGATIVO")
        else:
            print("   Sentimiento: NEUTRAL")

        print("\n✓ Test completado exitosamente")
    else:
        print("\n⚠ No se pudo completar el test (modelo no disponible)")
        print("Instala transformers con: pip install transformers torch")

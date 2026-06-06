# ==============================================================================
# PROJEKT: Zaawansowana analiza sentymentu i TF-IDF dla opinii o Starbucks
# ==============================================================================

# Wymagane pakiety ----
library(tm)
library(tidytext)
library(stringr)
library(wordcloud)
library(RColorBrewer)
library(ggplot2)
library(SnowballC)
library(SentimentAnalysis)
library(ggthemes)
library(tidyverse)
library(textdata)

# 1. Dane wejściowe ----
raw_data <- read.csv("reviews_data.csv", stringsAsFactors = FALSE, encoding = "UTF-8")

# Pobieramy same teksty do korpusu
text_data <- raw_data$Review

# Utworzenie korpusu dokumentów tekstowych (podejście z zajęć)
corpus <- VCorpus(VectorSource(text_data))

# 2. Przetwarzanie i oczyszczanie tekstu (Text Cleaning z pakietem 'tm') ----
# Zapewnienie kodowania w całym korpusie
corpus <- tm_map(corpus, content_transformer(function(x) iconv(x, to = "UTF-8", sub = "byte")))

# Funkcja do zamiany znaków na spację
toSpace <- content_transformer(function (x, pattern) gsub(pattern, " ", x))

# Usuwanie zbędnych znaków (zgodnie z kodem wykładowcy)
corpus <- tm_map(corpus, toSpace, "@")
corpus <- tm_map(corpus, toSpace, "@\\w+")
corpus <- tm_map(corpus, toSpace, "\\|")
corpus <- tm_map(corpus, toSpace, "[ \t]{2,}")
corpus <- tm_map(corpus, toSpace, "(s?)(f|ht)tp(s?)://\\S+\\b") # adresy URL
corpus <- tm_map(corpus, toSpace, "http\\w*")
corpus <- tm_map(corpus, toSpace, "/")
corpus <- tm_map(corpus, toSpace, "(RT|via)((?:\\b\\W*@\\w+)+)")
corpus <- tm_map(corpus, toSpace, "www")
corpus <- tm_map(corpus, toSpace, "~")

# Standardowa normalizacja
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, removePunctuation)

# Usunięcie specyficznych słów biznesowych, które nic nie wnoszą
corpus <- tm_map(corpus, removeWords, c("starbucks", "coffee", "store", "drink"))
corpus <- tm_map(corpus, stripWhitespace)

# Bezpieczny Stemming (rdzeniowanie) z pakietu SnowballC
# UWAGA: Celowo pomijamy stemCompletion z zajęć, ponieważ przy 
# dużym VCorpus niszczy on metadane dokumentów (błąd klasy "character").
corpus <- tm_map(corpus, stemDocument)

# 3. Macierz częstości TDM (Zwykła) i Eksploracyjna analiza danych ----
tdm <- TermDocumentMatrix(corpus)
tdm_m <- as.matrix(tdm)

# Zliczanie częstości słów
v <- sort(rowSums(tdm_m), decreasing = TRUE)
tdm_df <- data.frame(word = names(v), freq = v)

# Chmura słów (globalna)
set.seed(1234)
wordcloud(words = tdm_df$word, freq = tdm_df$freq, min.freq = 5, 
          max.words = 100, colors = brewer.pal(8, "Dark2"))
title("Globalna chmura słów (BoW)")

# 4. Macierz częstości TDM z wagami TF-IDF ----
tdm_tfidf <- TermDocumentMatrix(corpus,
                                control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE)))

tdm_tfidf_m <- as.matrix(tdm_tfidf)

# Zliczanie wag TF-IDF
v_tfidf <- sort(rowSums(tdm_tfidf_m), decreasing = TRUE)
tdm_tfidf_df <- data.frame(word = names(v_tfidf), freq = v_tfidf)

# Chmura słów oparta tylko na unikalnych markerach TF-IDF
set.seed(1234)
wordcloud(words = tdm_tfidf_df$word, freq = tdm_tfidf_df$freq, min.freq = 2, 
          max.words = 100, colors = brewer.pal(8, "Paired"))
title("Chmura najważniejszych słów (Wagi TF-IDF)")

# 5. Podział tekstu na równe segmenty o ustalonej długości (Analiza w czasie) ----
# Usunięcie pustych wierszy z surowego tekstu
non_empty_lines <- text_data[nzchar(text_data)]

# Połączenie wszystkich wierszy w jeden ciąg znaków
full_text <- paste(non_empty_lines, collapse = " ")
full_text <- gsub("\\s+", " ", full_text)

# Funkcja do dzielenia tekstu na segmenty o określonej długości
split_text_into_chunks <- function(text, chunk_size) {
  start_positions <- seq(1, nchar(text), by = chunk_size)
  chunks <- substring(text, start_positions, start_positions + chunk_size - 1)
  return(chunks)
}

# Podzielenie tekstu (segmenty po 150 znaków)
min_length <- 150
text_chunks <- split_text_into_chunks(full_text, min_length)
# 6. Analiza sentymentu przy użyciu pakietu SentimentAnalysis ----
sentiment <- analyzeSentiment(text_chunks)

# Pobranie sentymentu z 4 różnych słowników badawczych
df_all <- data.frame(sentence = 1:length(sentiment[,1]),
                     GI = sentiment$SentimentGI,     # General Inquirer
                     HE = sentiment$SentimentHE,     # Henry’s Financial dictionary
                     LM = sentiment$SentimentLM,     # Loughran-McDonald
                     QDAP = sentiment$SentimentQDAP) # Quantitative Discourse Analysis

# Usunięcie brakujących wartości (NA), aby wykres zadziałał
df_all <- df_all[complete.cases(df_all), ]
df_all <- df_all[!is.na(df_all$QDAP), ]

# 7. Wykresy przedstawiające ewolucję sentymentu w czasie ----
ggplot(df_all, aes(x=sentence, y=QDAP)) + 
  geom_smooth(color="red", se = FALSE) +
  geom_smooth(aes(x=sentence, y=GI), color="green", se = FALSE) +
  geom_smooth(aes(x=sentence, y=HE), color="blue", se = FALSE) +
  geom_smooth(aes(x=sentence, y=LM), color="orange", se = FALSE) +
  labs(x = "Oś czasu (Kolejne segmenty tekstu)", 
       y = "Sentyment (Wygładzony)",
       title = "Zmiana sentymentu klientów w czasie",
       subtitle = "Porównanie 4 słowników: Czerwony(QDAP), Zielony(GI), Niebieski(HE), Pomarańczowy(LM)") +
  theme_gdocs()

# ==============================================================================
# 8. Tradycyjna Analiza Sentymentu (Słownik Bing - Wykres słupkowy)
# ==============================================================================
tidy_tokeny <- tdm_df %>%
  rename(word = word, n = freq) %>%
  inner_join(get_sentiments("bing"))

word_counts_bing <- tidy_tokeny %>%
  filter(sentiment %in% c("positive", "negative")) %>%
  group_by(sentiment) %>%
  top_n(15, n) %>%
  ungroup() %>%
  mutate(word2 = factor(word, levels = rev(unique(word))))

ggplot(word_counts_bing, aes(x=word2, y=n, fill=sentiment)) + 
  geom_col(show.legend=FALSE) +
  facet_wrap(~sentiment, scales="free") +
  coord_flip() +
  labs(x = "Słowa", y = "Liczba", title = "Top słowa wg sentymentu (Słownik Bing)") +
  theme_gdocs() + 
  scale_fill_manual(values = c("dodgerblue4", "goldenrod1"))

rm(list = ls())

# Working dir
setwd("/Users/paulaarraizaarias/Documents/ucm/tfg/matematicas/prediction_project_v1")

# Load necessary files
function_files <- list.files(path = "processed_data/arima_processing/functions", pattern = "\\.R$", full.names = TRUE)
sapply(function_files, source)

# 1. Load data and obtain df
stock_name <- "ICEIB1Y"
file_path <- file.path(getwd(), "raw_data/stocks", paste0(stock_name, "_Close.csv"))
df <- read.csv(file_path, sep = ";", dec = ",")

result <- generate_ts(df)
ts_ret <- result$ts
df_full_ret <- result$df_full_ret

# 2. Outliers and differencing
diff_result <- check_differencing(df_full_ret, seasonal_period=365)
normal_diff_order <- diff_result$normal_diff_order
seasonal_diff_order <- diff_result$seasonal_diff_order
df_wo_outliers <- remove_outliers(df_full_ret)

# Intermediate step: divide between train and test
ts_ret <- df_full_ret$Return
train_size <- 0.8
n <- length(ts_ret)
train_index <- floor(train_size * n)
ts_train <- ts_ret[1:train_index]
ts_test <- ts_ret[(train_index + 1):n]

cat("Training on first", length(ts_train), "observations.\n")
cat("Testing on remaining", length(ts_test), "observations.\n")

# 3. Load data and obtain df
if (any(is.na(ts_ret))) {
  print("NA values in df")
} else 
{
  if (any(seasonal_diff_order, normal_diff_order) != 0) {
    print("Differentiation needed")
  } else {
    arima_result <- analyze_time_series(ts_train, max_p=3, max_d=normal_diff_order, max_q=3, max_r=3, max_s=3, garch_threshold = 0.6)
    best_p <- arima_result$p
    best_q <- arima_result$q
    best_r <- arima_result$r
    best_s <- arima_result$s
    
    # Now, start predicting
    # Assuming ts_ret is your time series data and you have obtained p, q, r, s from your model
    prediction_result <- evaluate_garch_model(ts_ret, p = best_p, q = best_q, r = best_r, s = best_s, train_size=train_size)
    test_length <- length(ts_ret) - prediction_result$train_index
    
    # Plot Actual vs Predicted Returns
    df <- data.frame(
      Time = seq(1, test_length),
      Actual = coredata(tail(ts_ret, test_length)),
      Predicted = prediction_result$predictions
    )
    ggplot_obj <- ggplot(df, aes(x = Time)) +
      geom_line(aes(y = Actual, color = "Actual")) +
      geom_line(aes(y = Predicted, color = "Predicted"), linetype = "dashed") +
      labs(title = "Actual vs Predicted Returns", x = "Time", y = "Returns") +
      theme_minimal() +
      scale_color_manual("", values = c("Actual" = "blue", "Predicted" = "red"))
    ggsave(filename = paste0("plots/arima_garch/", stock_name, "/actual_vs_predicted.png"), plot = ggplot_obj, width = 8, height = 6)
    
    # Save accuracy df
    accuracy <- evaluate_direction_accuracy(ts_ret, prediction_result$predictions, prediction_result$train_index, 
                                            p = best_p, q = best_q, r = best_r, s = best_s)
    filename <- paste0("prediction_results/arima_garch/", stock_name, "_garch_p", best_p, "_q", best_q, "_r", best_r, "_s", best_s, ".csv")
    write.csv(accuracy$df, file.path(getwd(), filename), row.names = FALSE)
  }
}

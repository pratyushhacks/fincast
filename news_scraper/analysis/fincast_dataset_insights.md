# FinCast Dataset Analysis — Consolidated Insights

_Generated: 2026-08-17 22:30_

## Dataset Overview
- Total articles: 8,773
- Date range: 2026-03-14 to 2026-07-31
- Unique tickers: 20
- Articles with price data: 7,322

## 1. Overall Price Bucket Distribution
| bucket_1d | count | pct |
|---|---|---|
| strong_down | 436 | 6.0 |
| down | 1420 | 19.4 |
| flat | 3172 | 43.3 |
| up | 1629 | 22.2 |
| strong_up | 665 | 9.1 |

## 1b. Price Bucket % per Ticker
| ticker | strong_down_pct | down_pct | flat_pct | up_pct | strong_up_pct | flat_flag |
|---|---|---|---|---|---|---|
| AAPL | 7.9 | 22.9 | 32.0 | 25.9 | 11.3 | OK |
| AMD | 7.2 | 18.4 | 40.8 | 16.4 | 17.1 | OK |
| AMZN | 10.8 | 16.4 | 39.4 | 24.0 | 9.5 | OK |
| BOTZ | nan | 28.2 | 35.9 | 23.1 | 12.8 | OK |
| DRIV | 6.2 | 22.3 | 35.4 | 26.9 | 9.2 | OK |
| GOOGL | 2.1 | 25.0 | 46.5 | 14.8 | 11.6 | OK |
| INTC | 1.0 | 26.8 | 41.2 | 20.3 | 10.7 | OK |
| META | 5.6 | 18.6 | 54.3 | 15.7 | 5.7 | OK |
| MSFT | 7.4 | 18.2 | 42.3 | 22.8 | 9.4 | OK |
| NVDA | 3.6 | 18.2 | 52.7 | 19.2 | 6.3 | OK |
| SKYY | 4.0 | 11.6 | 39.9 | 27.3 | 17.2 | OK |
| SOXX | 5.7 | 15.4 | 34.3 | 28.6 | 16.1 | OK |
| SPY | 5.1 | 10.7 | 55.2 | 23.4 | 5.6 | OK |
| TLT | 10.9 | 19.4 | 32.0 | 30.6 | 7.0 | OK |
| TSLA | 10.5 | 20.8 | 45.6 | 13.0 | 10.0 | OK |
| USO | 9.8 | 17.8 | 25.4 | 38.1 | 8.9 | OK |
| UUP | 3.6 | 15.5 | 41.1 | 35.7 | 4.2 | OK |
| XLF | 4.8 | 17.2 | 36.6 | 33.0 | 8.4 | OK |
| XLK | 3.3 | 10.0 | 30.0 | 43.3 | 13.3 | OK |
| XLV | 0.7 | 40.1 | 39.4 | 13.5 | 6.4 | OK |

## 1c. Move Magnitude Distribution
| magnitude | count | pct |
|---|---|---|
| minimal | 3374 | 46.1 |
| small | 1659 | 22.7 |
| moderate | 1762 | 24.1 |
| large | 527 | 7.2 |

## 2. Sentiment vs Price — Confusion Matrix
| sent_signal | down | flat | up | Total |
|---|---|---|---|---|
| bearish | 265 | 481 | 366 | 1112 |
| bullish | 912 | 1528 | 1104 | 3544 |
| neutral | 679 | 1163 | 824 | 2666 |
| Total | 1856 | 3172 | 2294 | 7322 |

## 2b. Mismatch Rate per Ticker
| ticker | mismatch_rate | mismatch_pct | flag |
|---|---|---|---|
| DRIV | 0.24615384615384617 | 24.6 | OK |
| AMD | 0.23026315789473684 | 23.0 | OK |
| AAPL | 0.21578099838969403 | 21.6 | OK |
| AMZN | 0.21271393643031786 | 21.3 | OK |
| GOOGL | 0.20601851851851852 | 20.6 | OK |
| XLV | 0.19858156028368795 | 19.9 | OK |
| META | 0.19052987598647125 | 19.1 | OK |
| USO | 0.19047619047619047 | 19.0 | OK |
| INTC | 0.18556701030927836 | 18.6 | OK |
| TLT | 0.18309859154929578 | 18.3 | OK |
| MSFT | 0.17985611510791366 | 18.0 | OK |
| UUP | 0.17261904761904762 | 17.3 | OK |
| NVDA | 0.17149478563151796 | 17.1 | OK |
| SOXX | 0.16071428571428573 | 16.1 | OK |
| TSLA | 0.15789473684210525 | 15.8 | OK |
| XLF | 0.1013215859030837 | 10.1 | OK |
| XLK | 0.1 | 10.0 | OK |
| SKYY | 0.09090909090909091 | 9.1 | OK |
| BOTZ | 0.07692307692307693 | 7.7 | OK |
| SPY | 0.07084019769357495 | 7.1 | OK |

## 3. Edge Cases — Sentiment/Price Mismatch Counts
- Total mismatches: 1278 (17.5% of labeled articles)
- Bullish sentiment but price dropped: 912
- Bearish sentiment but price rose: 366
- Largest mismatched move: 14.52% (1-day change)
- Tickers with most mismatches: META (169), NVDA (148), AAPL (134), MSFT (100), GOOGL (89)

## 4. Stock vs Benchmark Correlation
| ticker | benchmark | r | p | n |
|---|---|---|---|---|
| AMD | QQQ | 0.885 | 0.0 | 304 |
| TSLA | QQQ | 0.639 | 0.0 | 399 |
| INTC | QQQ | 0.624 | 0.0 | 291 |
| NVDA | QQQ | 0.614 | 0.0 | 863 |
| AMZN | QQQ | 0.44 | 0.0 | 409 |
| GOOGL | QQQ | 0.417 | 0.0 | 432 |
| META | QQQ | 0.386 | 0.0 | 887 |
| MSFT | QQQ | 0.239 | 0.0 | 556 |
| AAPL | QQQ | 0.228 | 0.0 | 621 |

## 5. Weekly Macro Sentiment vs Avg Stock Change
| week_end | macro_sentiment | avg_stock_change | macro_signal |
|---|---|---|---|
| 2026-03-14 (Sat) | 3.2 | 1.7207 | neutral |
| 2026-03-21 (Sat) | 3.3468 | -0.186 | neutral |
| 2026-03-28 (Sat) | 3.2931 | -0.9826 | neutral |
| 2026-04-04 (Sat) | 3.2523 | 0.4207 | neutral |
| 2026-04-11 (Sat) | 3.2626 | 1.7181 | neutral |
| 2026-04-18 (Sat) | 3.3122 | 0.1431 | neutral |
| 2026-04-25 (Sat) | 3.3492 | 0.8946 | neutral |
| 2026-05-02 (Sat) | 3.1986 | 1.3439 | neutral |
| 2026-05-09 (Sat) | 3.305 | 1.0625 | neutral |
| 2026-05-16 (Sat) | 3.3519 | -0.4314 | neutral |
| 2026-05-23 (Sat) | 3.2756 | 0.1253 | neutral |
| 2026-05-30 (Sat) | 3.3571 | 0.1137 | neutral |
| 2026-06-06 (Sat) | 3.3194 | -0.898 | neutral |
| 2026-06-13 (Sat) | 3.1284 | 0.6838 | neutral |
| 2026-06-20 (Sat) | 3.4132 | 0.3457 | neutral |
| 2026-06-27 (Sat) | 3.1781 | 0.1521 | neutral |
| 2026-07-04 (Sat) | 3.3548 | 1.9364 | neutral |
| 2026-07-11 (Sat) | 3.3218 | 0.5191 | neutral |
| 2026-07-18 (Sat) | 3.3829 | -0.4264 | neutral |
| 2026-07-25 (Sat) | 3.4103 | -1.6594 | neutral |
| 2026-08-01 (Sat) | 3.3041 | -0.0867 | neutral |

## 6. Sector Sentiment vs Constituent Stock Correlation
| sector_etf | sector_name | stock | r | p | n_days | significant |
|---|---|---|---|---|---|---|
| SPY | S&P500 | MSFT | 0.218 | 0.0704 | 70 | No |
| BOTZ | AI/Robotics | NVDA | 0.087 | 0.6467 | 30 | No |
| SPY | S&P500 | AAPL | 0.02 | 0.8699 | 70 | No |
| SPY | S&P500 | AMD | 0.015 | 0.916 | 55 | No |
| SOXX | Semiconductor | INTC | 0.012 | 0.9306 | 52 | No |
| SOXX | Semiconductor | NVDA | 0.004 | 0.972 | 64 | No |
| SPY | S&P500 | GOOGL | -0.01 | 0.9333 | 66 | No |
| SPY | S&P500 | TSLA | -0.038 | 0.7574 | 69 | No |
| DRIV | EV | TSLA | -0.038 | 0.783 | 56 | No |
| SPY | S&P500 | INTC | -0.042 | 0.7373 | 65 | No |
| SPY | S&P500 | AMZN | -0.096 | 0.4167 | 73 | No |
| SPY | S&P500 | META | -0.146 | 0.2029 | 78 | No |
| SOXX | Semiconductor | AMD | -0.164 | 0.2934 | 43 | No |
| SPY | S&P500 | NVDA | -0.204 | 0.0878 | 71 | No |

## 7. Volatility Regime vs Sentiment Predictive Power
| ticker | regime | r | p | n |
|---|---|---|---|---|
| AAPL | high_vol | -0.037 | 0.5134 | 314 |
| AAPL | low_vol | -0.079 | 0.1711 | 304 |
| AMD | high_vol | 0.036 | 0.6498 | 160 |
| AMD | low_vol | -0.051 | 0.5404 | 144 |
| AMZN | high_vol | 0.115 | 0.0985 | 208 |
| AMZN | low_vol | 0.16 | 0.0234 | 201 |
| BOTZ | high_vol | -0.043 | 0.8588 | 20 |
| BOTZ | low_vol | 0.359 | 0.1314 | 19 |
| DRIV | high_vol | -0.111 | 0.367 | 68 |
| DRIV | low_vol | -0.022 | 0.863 | 62 |
| GOOGL | high_vol | 0.037 | 0.5875 | 220 |
| GOOGL | low_vol | -0.086 | 0.2157 | 208 |
| INTC | high_vol | 0.202 | 0.0132 | 150 |
| INTC | low_vol | 0.094 | 0.2693 | 141 |
| META | high_vol | -0.032 | 0.4947 | 471 |
| META | low_vol | -0.117 | 0.019 | 404 |
| MSFT | high_vol | -0.022 | 0.7148 | 283 |
| MSFT | low_vol | 0.05 | 0.4182 | 266 |
| NVDA | high_vol | -0.014 | 0.7743 | 439 |
| NVDA | low_vol | 0.025 | 0.6107 | 424 |
| SKYY | high_vol | 0.09 | 0.3592 | 106 |
| SKYY | low_vol | 0.175 | 0.0946 | 92 |
| SOXX | high_vol | -0.151 | 0.0728 | 142 |
| SOXX | low_vol | 0.249 | 0.0032 | 138 |
| SPY | high_vol | 0.024 | 0.6638 | 320 |
| SPY | low_vol | 0.046 | 0.441 | 287 |
| TLT | high_vol | 0.059 | 0.4853 | 142 |
| TLT | low_vol | -0.076 | 0.3708 | 142 |
| TSLA | high_vol | 0.1 | 0.1568 | 203 |
| TSLA | low_vol | 0.065 | 0.364 | 195 |
| USO | high_vol | 0.093 | 0.2399 | 162 |
| USO | low_vol | -0.089 | 0.2722 | 153 |
| UUP | high_vol | -0.068 | 0.5372 | 84 |
| UUP | low_vol | -0.071 | 0.5216 | 84 |
| XLF | high_vol | -0.054 | 0.5637 | 116 |
| XLF | low_vol | 0.194 | 0.0408 | 111 |
| XLK | high_vol | -0.222 | 0.4273 | 15 |
| XLK | low_vol | 0.125 | 0.6567 | 15 |
| XLV | high_vol | -0.062 | 0.4491 | 151 |
| XLV | low_vol | -0.045 | 0.6128 | 130 |

## 8. Relevance Tier vs Signal Strength
| relevance | n | r | p | avg_abs_change_1d | significant |
|---|---|---|---|---|---|
| medium | 5427.0 | 0.037 | 0.0066 | 1.698 | Yes |
| high | 1895.0 | 0.038 | 0.1019 | 2.292 | No |

## 9. Per-Ticker Readiness
| ticker | category | total_articles | unique_days | avg_sentiment | avg_trailing_vol | flat_pct | strong_down_pct | strong_up_pct | bucket_entropy | imbalance_flag | readiness |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AAPL | company | 621 | 80 | 3.52 | 1.54 | 32.0 | 7.9 | 11.3 | 0.93 | OK | READY 200+ |
| TLT | sector/macro | 284 | 65 | 3.19 | 0.58 | 32.0 | 10.9 | 7.0 | 0.92 | OK | READY 200+ |
| AMZN | company | 409 | 83 | 3.75 | 1.86 | 39.4 | 10.8 | 9.5 | 0.91 | OK | READY 200+ |
| AMD | company | 304 | 63 | 4.08 | 5.13 | 40.8 | 7.2 | 17.1 | 0.91 | OK | READY 200+ |
| USO | sector/macro | 315 | 70 | 3.6 | 4.23 | 25.4 | 9.8 | 8.9 | 0.91 | OK | READY 200+ |
| SOXX | sector/macro | 280 | 76 | 3.71 | 2.99 | 34.3 | 5.7 | 16.1 | 0.91 | OK | READY 200+ |
| DRIV | sector/macro | 130 | 58 | 3.58 | 2.17 | 35.4 | 6.2 | 9.2 | 0.9 | OK | MED 50-200 |
| MSFT | company | 556 | 80 | 3.41 | 1.91 | 42.3 | 7.4 | 9.4 | 0.88 | OK | READY 200+ |
| TSLA | company | 399 | 78 | 3.2 | 3.06 | 45.6 | 10.5 | 10.0 | 0.88 | OK | READY 200+ |
| SKYY | sector/macro | 198 | 65 | 3.43 | 2.11 | 39.9 | 4.0 | 17.2 | 0.87 | OK | MED 50-200 |
| XLF | sector/macro | 227 | 70 | 3.1 | 0.97 | 36.6 | 4.8 | 8.4 | 0.86 | OK | READY 200+ |
| XLK | sector/macro | 30 | 22 | 3.13 | 1.66 | 30.0 | 3.3 | 13.3 | 0.83 | OK | LOW <50 |
| BOTZ | sector/macro | 39 | 30 | 3.21 | 1.88 | 35.9 | 0.0 | 12.8 | 0.82 | OK | LOW <50 |
| GOOGL | company | 432 | 73 | 3.63 | 2.22 | 46.5 | 2.1 | 11.6 | 0.82 | OK | READY 200+ |
| INTC | company | 291 | 70 | 3.95 | 5.68 | 41.2 | 1.0 | 10.7 | 0.82 | OK | READY 200+ |
| UUP | sector/macro | 168 | 56 | 2.87 | 0.35 | 41.1 | 3.6 | 4.2 | 0.79 | OK | MED 50-200 |
| NVDA | company | 863 | 78 | 3.73 | 2.48 | 52.7 | 3.6 | 6.3 | 0.78 | OK | READY 200+ |
| META | company | 887 | 90 | 3.25 | 2.78 | 54.3 | 5.6 | 5.7 | 0.78 | OK | READY 200+ |
| SPY | sector/macro | 607 | 92 | 3.06 | 0.89 | 55.2 | 5.1 | 5.6 | 0.76 | OK | READY 200+ |
| XLV | sector/macro | 282 | 72 | 3.34 | 1.06 | 39.4 | 0.7 | 6.4 | 0.75 | OK | READY 200+ |

## 10. Overall Readiness Summary
- **total_examples**: 7322
- **total_ticker_groups**: 20
- **ready_tickers**: 15
- **medium_tickers**: 3
- **low_tickers**: 2
- **high_flat_tickers**: 0
- **low_entropy_tickers**: 0
- **low_sample_tickers**: 2
- **overall_flat_pct**: 43.3
- **overall_bucket_entropy**: 0.87
- **neutral_sentiment_pct**: 37.2
- **dataset_mismatch_pct**: 16.715
- **ready_to_train**: True
- **recommendation**: Proceed with fine-tuning

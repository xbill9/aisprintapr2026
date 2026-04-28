
import termplotlib as tpl
import numpy as np

# Data from the benchmark results
ttft_labels = ["Mean TTFT", "Median TTFT", "P99 TTFT"]
ttft_values = np.array([413.79, 408.13, 569.48])

tpot_labels = ["Mean TPOT", "Median TPOT", "P99 TPOT"]
tpot_values = np.array([61.71, 61.89, 62.21])

itl_labels = ["Mean ITL", "Median ITL", "P99 ITL"]
itl_values = np.array([61.71, 61.98, 66.36])

# Create the plots
print("--- Time to First Token (ms) ---")
fig1 = tpl.figure()
fig1.barh(ttft_values, ttft_labels, force_ascii=False)
fig1.show()

print("\n--- Time per Output Token (ms) ---")
fig2 = tpl.figure()
fig2.barh(tpot_values, tpot_labels, force_ascii=False)
fig2.show()

print("\n--- Inter-token Latency (ms) ---")
fig3 = tpl.figure()
fig3.barh(itl_values, itl_labels, force_ascii=False)
fig3.show()

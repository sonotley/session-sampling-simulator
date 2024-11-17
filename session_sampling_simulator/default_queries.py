queries_yaml = """
---
queries:
 - id: 1
   mean_duration: 40
   duration_spread: 0
   target_periodicity: 1000
   wait_state_ratios:
    1: 3
    2: 2
    3: 1
   duration_distribution: exponential
 - id: 2
   mean_duration: 25
   duration_spread: 0
   target_periodicity: 1000
   wait_state_ratios:
     1: 3
     2: 2
     3: 1
   duration_distribution: exponential
 - id: 3
   mean_duration: 1500
   duration_spread: 1000
   target_periodicity: 50000
   wait_state_ratios:
     1: 3
     2: 2
     3: 1
 - id: 4
   mean_duration: 1500
   duration_spread: 1000
   target_periodicity: 50000
   wait_state_ratios:
     1: 3
     2: 2
     3: 1
 - id: 5
   mean_duration: 1500
   duration_spread: 1000
   target_periodicity: 50000
   wait_state_ratios:
     1: 3
     2: 2
     3: 1
"""

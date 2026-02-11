# Copyright 2026 Nearby Computing S.L.
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.colors as mcolors

# Function to generate a more diverse user and traffic distribution
def generate_traffic_distribution(num_users, num_apps):
    # Define user groups with different traffic patterns
    high_traffic_users = int(num_users * 0.1)  # 10% of users with high traffic
    low_traffic_users = num_users - high_traffic_users  # Remaining users with low traffic

    # Assign traffic demands based on user groups
    high_traffic_mbps = np.random.randint(15, 20, size=high_traffic_users)  # High traffic users (15-20 Mbps)
    low_traffic_mbps = np.random.randint(2, 8, size=low_traffic_users)  # Low traffic users (2-8 Mbps)

    # Combine traffic demands
    mbps_per_user = np.concatenate((high_traffic_mbps, low_traffic_mbps))
    np.random.shuffle(mbps_per_user)  # Shuffle to mix high and low traffic users

    # Randomly assign users to each application, ensuring diversity in the number of users per app
    users_per_app = np.random.randint(1, num_users // 2, size=num_apps)
    users_per_app = users_per_app / users_per_app.sum() * num_users  # Normalize to sum to num_users

    # Calculate traffic per app, where some apps may have high users and low traffic or low users and high traffic
    app_traffic = []
    user_idx = 0
    for i in range(num_apps):
        app_user_count = int(users_per_app[i])
        app_traffic.append(sum(mbps_per_user[user_idx:user_idx + app_user_count]))
        user_idx += app_user_count

    return users_per_app, app_traffic


# Function to analyze migration strategies
def analyze_migration_strategies(strategy, num_users, min_replica_mbps, num_iterations=1000):
    # Configuration
    num_apps = 10
    server_capacity = 16  # CPU cores
    min_replica_cpu = 2
    max_replica_cpu = 3
    weight_factor = 1  # High weight factor

    affected_percentages = []

    for _ in range(num_iterations):
        # Generate traffic distribution with random users and traffic per app
        users_per_app, app_traffic = generate_traffic_distribution(num_users, num_apps)

        # Calculate CPU requirements based on the generated traffic
        cpu_requirements = []
        for mbps in app_traffic:
            replicas = int(np.ceil(mbps / min_replica_mbps))
            cpu_per_replica = np.random.randint(min_replica_cpu, max_replica_cpu + 1)
            total_cpu = replicas * cpu_per_replica
            cpu_requirements.append(total_cpu)

        # Migration logic
        apps = list(range(num_apps))  # Application indices
        total_cpu_used = sum(cpu_requirements)
        apps_to_migrate = []

        if total_cpu_used > server_capacity:
            if strategy == "random":
                np.random.shuffle(apps)
            elif strategy == "cpu-based-high-cpu":
                apps.sort(key=lambda i: cpu_requirements[i], reverse=True)  # Highest CPU first
            elif strategy == "cpu-based-low-cpu":
                apps.sort(key=lambda i: cpu_requirements[i])  # Lowest CPU first
            elif strategy == "network-aware":
                apps.sort(key=lambda i: (users_per_app[i], cpu_requirements[i]))  # Fewest users and weighted CPU

            migrated_cpu = 0
            for app in apps:
                if total_cpu_used - migrated_cpu <= server_capacity:
                    break
                migrated_cpu += cpu_requirements[app]
                apps_to_migrate.append(app)

        # Calculate percentage of affected users
        migrated_users = sum([users_per_app[app] for app in apps_to_migrate])
        percentage_affected = (migrated_users / num_users) * 100
        affected_percentages.append(percentage_affected)

    # Return the average percentage over multiple iterations
    return np.mean(affected_percentages)


# Set seed for reproducibility
np.random.seed(42)

# Define different user scenarios
num_users_list = [10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400]
strategies = ["random", "cpu-based-high-cpu", "cpu-based-low-cpu", "network-aware"]

# Prepare result storage for two traffic scenarios
results_100mbps = {strategy: [] for strategy in strategies}
results_500mbps = {strategy: [] for strategy in strategies}

# Run experiments for each number of users with two different traffic configurations
for num_users in num_users_list:
    for strategy in strategies:
        # For 100 Mbps traffic scenario
        affected_percentage_100mbps = analyze_migration_strategies(strategy, num_users, min_replica_mbps=100)
        results_100mbps[strategy].append(affected_percentage_100mbps)
        
        # For 500 Mbps traffic scenario
        affected_percentage_500mbps = analyze_migration_strategies(strategy, num_users, min_replica_mbps=500)
        results_500mbps[strategy].append(affected_percentage_500mbps)

# Save results to CSV file
with open('migration_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Number of Users', 'Strategy', 'Traffic Scenario', 'Affected Percentage'])
    
    for num_users in num_users_list:
        for strategy in strategies:
            writer.writerow([num_users, strategy, '100 Mbps', results_100mbps[strategy][num_users_list.index(num_users)]])
            writer.writerow([num_users, strategy, '500 Mbps', results_500mbps[strategy][num_users_list.index(num_users)]])

# Define colors for each strategy using Matplotlib's default "tab10" palette
tableau_colors = list(mcolors.TABLEAU_COLORS.values())
colors = {
    "random": tableau_colors[0],  # Blue
    "cpu-based-high-cpu": tableau_colors[1],  # Orange
    "cpu-based-low-cpu": tableau_colors[2],  # Green
    "network-aware": tableau_colors[3]  # Red
}

# Plot results for both traffic scenarios
plt.figure(figsize=(10, 6))

# Custom legend labels
custom_labels = {
    "random": "Random-SM",
    "cpu-based-high-cpu": "CPU-based-High-SM",
    "cpu-based-low-cpu": "CPU-based-Low-SM",
    "network-aware": "NASO"
}

# Plot for 100 Mbps scenario
for strategy in strategies:
    plt.plot(num_users_list, results_100mbps[strategy], marker='o', label=f'100 Mbps - {custom_labels[strategy]}', color=colors[strategy])

# Plot for 500 Mbps scenario
for strategy in strategies:
    plt.plot(num_users_list, results_500mbps[strategy], marker='x', linestyle='--', label=f'500 Mbps - {custom_labels[strategy]}', color=colors[strategy])

plt.xlabel('Number of users in a region', fontsize=18)  # Increase font size for x-axis label
plt.ylabel('UASM (%)', fontsize=18)  # Increase font size for y-axis label
plt.xticks(fontsize=16)  # Increase font size for x-axis tick values
plt.yticks(fontsize=16)  # Increase font size for y-axis tick values
#plt.title('Impact of Migration on Users Across Strategies for Different Traffic Scenarios')
plt.legend(loc='upper left', fontsize=10)  # Increase font size for legend
#plt.ylim(0, max(max(results_100mbps[strategy]) for strategy in strategies) + 10)  # Adjust y-axis to prevent legend overlap
plt.ylim(0, 100)  # Set y-axis limits to include 100
plt.grid(True)
plt.savefig('users_affected_migration_comparison_percentage.pdf')
plt.show()
### Permet de visualiser les distributions de dirichlet

from ByrdLab import attack
from ByrdLab.aggregation import C_mean
from ByrdLab.centraliedAlgorithm import CMomentum_under_DPA
from ByrdLab.environment import Dist_Dataset_Opt_Env
from ByrdLab.library.dataset import DistributedDataSets_over_honest_and_byz_nodes, cifar10, mnist
from ByrdLab.library.partition import TrivalPartition, DirichletPartition_a
from ByrdLab.tasks.softmaxRegression import softmaxRegressionTask
import matplotlib.pyplot as plt
import numpy as np
from ByrdLab.library.partition import DirichletPartition_a, DirichletPartition_b, DirichletPartition_c, DirichletPartition_d, DirichletPartition_e, DirichletPartition_f
alphas = [100, 10, 1, 0.1, 0.01, 0.001]  # Dirichlet 'a' values from a to f
partition_classes = [
    DirichletPartition_a,
    DirichletPartition_b,
    DirichletPartition_c,
    DirichletPartition_d,
    DirichletPartition_e,
    DirichletPartition_f
]

fig, axes = plt.subplots(1, len(alphas), figsize=(4 * len(alphas), 6), sharey=True)

seed = 1


for idx, (a, partition_cls) in enumerate(zip(alphas, partition_classes)):
    data_package = cifar10()
    task = softmaxRegressionTask(data_package, batch_size=32)
    step_agg = 1
    lr_ctrl = None
    fix_seed = True
    node_size = 10
    byzantine_size = 1
    all_nodes = list(range(node_size))
    honest_nodes = list(range(node_size - byzantine_size))
    byzantine_nodes = [node for node in all_nodes if node not in honest_nodes]
    aggregation = C_mean(honest_nodes, byzantine_nodes)

    env = CMomentum_under_DPA(
        aggregation=aggregation,
        honest_nodes=honest_nodes,
        byzantine_nodes=byzantine_nodes,
        attack=attack,
        step_agg=step_agg,
        weight_decay=task.weight_decay,
        data_package=task.data_package,
        model=task.model,
        loss_fn=task.loss_fn,
        test_fn=task.test_fn,
        initialize_fn=task.initialize_fn,
        get_train_iter=task.get_train_iter,
        get_test_iter=task.get_test_iter,
        partition_cls=partition_cls,
        lr_ctrl=lr_ctrl,
        fix_seed=fix_seed,
        seed=seed,
        **task.super_params
    )

    dist_train_set = env.dist_train_set
    class_colors = plt.cm.get_cmap('tab10')
    y_positions = np.arange(node_size)
    class_counts_per_node = []

    for node in all_nodes:
        node_dataset = dist_train_set[node]
        node_labels = np.array([label for _, label in node_dataset])
        class_counts = [np.sum(node_labels == class_id) for class_id in range(data_package.num_classes)]
        class_counts_per_node.append(class_counts)

    class_counts_per_node = np.array(class_counts_per_node)
    normalized_counts = class_counts_per_node / class_counts_per_node.sum(axis=1, keepdims=True)

    left = np.zeros(node_size)
    ax = axes[idx]
    for class_id in range(data_package.num_classes):
        ax.barh(y_positions, normalized_counts[:, class_id], left=left,
                color=class_colors(class_id), label=f'Class {class_id}' if idx == 0 else None)
        left += normalized_counts[:, class_id]

    ax.set_title(f'{partition_cls.__name__} (a={a})')
    ax.set_xlabel('Proportion')
    # Indicate honest and byzantine nodes on y-axis
    yticks_labels = []
    for node in all_nodes:
        if node in honest_nodes:
            yticks_labels.append(f'Node {node} (Honest)')
        else:
            yticks_labels.append(f'Node {node} (Byzantine)')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(yticks_labels)
    # if idx == 0:
    #     ax.set_ylabel('Nodes')
    #     ax.legend(title='Classes')
    # else:
    #     # Affiche uniquement pour les noeuds byzantins
    #     ax.set_yticks([node for node in byzantine_nodes])
    #     ax.set_yticklabels([f'Node {node} (Byzantine)' for node in byzantine_nodes])
    #     ax.set_yticks([])

plt.tight_layout()
plt.savefig(f'draw_fig/pic/distribution_plot_dirichlet_a_to_f_{seed}_{data_package.__class__.__name__}.png')
plt.show()

import time
import random
import matplotlib.pyplot as plt
from tabulate import tabulate

class BTreeNode:
    def __init__(self, t, leaf=False):
        self.t = t
        self.keys = []
        self.children = []
        self.leaf = leaf

    def __str__(self, level=0, indent="    "):
        s = level * indent + str(self.keys) + "\n"
        for child in self.children:
            s += child.__str__(level + 1, indent)
        return s

    def from_string(self, s):
        # Simplified function to reconstruct a BTreeNode from a string
        parts = s.split('\n')
        self.keys = list(map(int, parts[0].strip('[]').split(',')))
        if len(parts) > 1 and parts[1].strip():
            self.children = [BTreeNode(self.t, leaf=True) for _ in range(len(parts[1].split('|')))]
            for i, child_str in enumerate(parts[1].split('|')):
                self.children[i].from_string(child_str)

class BTree:
    def __init__(self, t):
        self.root = BTreeNode(t, leaf=True)
        self.t = t

    def search(self, node, key):
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and key == node.keys[i]:
            return node.keys[i]

        if node.leaf:
            return None
        else:
            return self.search(node.children[i], key)

    def insert(self, key):
        root = self.root
        if len(root.keys) == 2 * self.t - 1:
            new_node = BTreeNode(self.t)
            self.root = new_node
            new_node.children.append(root)
            self._split_child(new_node, 0)
            self._insert_non_full(new_node, key)
        else:
            self._insert_non_full(root, key)

    def _insert_non_full(self, node, key):
        i = len(node.keys) - 1
        if node.leaf:
            node.keys.append(0)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                i -= 1
            node.keys[i + 1] = key
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if len(node.children[i].keys) == 2 * self.t - 1:
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key)

    def _split_child(self, parent, i):
        t = self.t
        node = parent.children[i]
        new_node = BTreeNode(t, leaf=node.leaf)
        parent.children.insert(i + 1, new_node)
        parent.keys.insert(i, node.keys[t - 1])

        new_node.keys = node.keys[t:(2 * t - 1)]
        node.keys = node.keys[0:t - 1]

        if not node.leaf:
            new_node.children = node.children[t:(2 * t)]
            node.children = node.children[0:t]

    def delete(self, key):
        self._delete(self.root, key)
        if len(self.root.keys) == 0:
            if len(self.root.children) > 0:
                self.root = self.root.children[0]
            else:
                self.root = None

    def _delete(self, node, key):
        t = self.t
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and key == node.keys[i]:
            if node.leaf:
                del node.keys[i]
            else:
                self._delete_internal_node(node, key, i)
        elif not node.leaf:
            self._delete_non_leaf(node, key, i)
        else:
            return None

    def _delete_internal_node(self, node, key, i):
        t = self.t
        if len(node.children[i].keys) >= t:
            node.keys[i] = self._get_predecessor(node, i)
            self._delete(node.children[i], node.keys[i])
        elif i + 1 < len(node.children) and len(node.children[i + 1].keys) >= t:
            node.keys[i] = self._get_successor(node, i)
            self._delete(node.children[i + 1], node.keys[i])
        elif i + 1 < len(node.children):
            self._merge_nodes(node, i)
            self._delete(node.children[i], key)
        else:
            print(f"Cannot delete key {key}. Node does not have enough children.")

    def _delete_non_leaf(self, node, key, i):
        t = self.t
        if len(node.children[i].keys) >= t:
            self._delete(node.children[i], key)
        elif i + 1 < len(node.children) and len(node.children[i + 1].keys) >= t:
            self._delete(node.children[i + 1], key)
        else:
            if i + 1 < len(node.children):
                self._merge_nodes(node, i)
                self._delete(node.children[i], key)
            else:
                print(f"Cannot delete key {key}. Node does not have enough children.")

    def _get_predecessor(self, node, i):
        current = node.children[i]
        while not current.leaf:
            current = current.children[-1]
        return current.keys[-1]

    def _get_successor(self, node, i):
        current = node.children[i + 1]
        while not current.leaf:
            current = current.children[0]
        return current.keys[0]

    def _merge_nodes(self, node, i):
        child = node.children[i]
        sibling = node.children[i + 1]
        child.keys.append(node.keys[i])
        child.keys.extend(sibling.keys)
        if not child.leaf:
            child.children.extend(sibling.children)
        del node.keys[i]
        del node.children[i + 1]

    def update(self, old_key, new_key):
        if self.search(self.root, old_key):
            self.delete(old_key)
            self.insert(new_key)
        else:
            print(f"Key {old_key} not found!")

    def display(self):
        if self.root:
            print(self.root)

    def save_to_file(self, filename):
        with open(filename, "w") as f:
            f.write(str(self.root))

    def save_tree(self, filename_prefix, size):
        self.save_to_file(f"{filename_prefix}_size_{size}.txt")

    def load_from_file(self, filename):
        with open(filename, "r") as f:
            content = f.read()
            self.root = BTreeNode(self.t)
            self.root.from_string(content)


def generate_random_data(size):
    return random.sample(range(1, size * 10), size)


def performance_test(btree, data, filename_prefix, size):
    insert_times = []
    search_times = []
    update_times = []
    delete_times = []

    # Teste de inserção
    print("Testando inserção...")
    start_time = time.time()
    for value in data:
        btree.insert(value)
    insert_time = time.time() - start_time
    insert_times.append(insert_time)
    print(f"Tempo de inserção: {insert_time:.4f} segundos")

    # Salvar a árvore após inserção
    btree.save_tree(filename_prefix, size)

    # Teste de busca (leitura)
    print("Testando busca...")
    start_time = time.time()
    for value in data:
        btree.search(btree.root, value)
    search_time = time.time() - start_time
    search_times.append(search_time)
    print(f"Tempo de busca: {search_time:.4f} segundos")

    # Teste de atualização
    print("Testando atualização...")
    start_time = time.time()
    if len(data) > 0:
        btree.update(data[0], data[0] + 100)
    update_time = time.time() - start_time
    update_times.append(update_time)
    print(f"Tempo de atualização: {update_time:.4f} segundos")

    # Salvar a árvore após atualização
    btree.save_tree(f"{filename_prefix}_updated", size)

    # Teste de exclusão
    print("Testando exclusão...")
    start_time = time.time()
    if len(data) > 0:
        btree.delete(data[0])
    delete_time = time.time() - start_time
    delete_times.append(delete_time)
    print(f"Tempo de exclusão: {delete_time:.4f} segundos")

    return insert_times, search_times, update_times, delete_times


def plot_performance(insert_times, search_times, update_times, delete_times, sizes):
    plt.figure(figsize=(12, 8))
    plt.plot(sizes, insert_times, label="Inserção", marker='o')
    plt.plot(sizes, search_times, label="Busca", marker='o')
    plt.plot(sizes, update_times, label="Atualização", marker='o')
    plt.plot(sizes, delete_times, label="Exclusão", marker='o')
    plt.xlabel("Tamanho da Árvore")
    plt.ylabel("Tempo (segundos)")
    plt.title("Desempenho da Árvore B")
    plt.legend()
    plt.grid(True)
    plt.show()


def display_results(sizes, insert_times, search_times, update_times, delete_times):
    table = []
    for size, insert, search, update, delete in zip(sizes, insert_times, search_times, update_times, delete_times):
        table.append([size, f"{insert:.4f}", f"{search:.4f}", f"{update:.4f}", f"{delete:.4f}"])

    headers = ["Tamanho", "Inserção", "Busca", "Atualização", "Exclusão"]
    print(tabulate(table, headers=headers, tablefmt="grid"))


def main():
    sizes = [100, 500, 1000, 5000, 10000]
    insert_times = []
    search_times = []
    update_times = []
    delete_times = []

    for size in sizes:
        data = generate_random_data(size)
        btree = BTree(t=3)  # Ordem da árvore B
        print(f"Testando árvore de tamanho {size}...")
        ins, sea, up, del_ = performance_test(btree, data, "btree", size)
        insert_times.extend(ins)
        search_times.extend(sea)
        update_times.extend(up)
        delete_times.extend(del_)

    plot_performance(insert_times, search_times, update_times, delete_times, sizes)
    display_results(sizes, insert_times, search_times, update_times, delete_times)

    # Exemplo de leitura da árvore do arquivo
    print("Lendo a árvore do arquivo...")
    btree_loaded = BTree(t=3)
    btree_loaded.load_from_file("btree_size_10000.txt")  # Substitua pelo nome do arquivo apropriado
    btree_loaded.display()


if __name__ == "__main__":
    main()

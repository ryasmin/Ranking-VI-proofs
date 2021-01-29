import numpy as np
import math
import copy


class WeakOrder:
    def __init__(self, wo_input):

        if isinstance(wo_input, int):
            self.num_bucs = wo_input
            self.bucs = [[] for i in range(self.num_bucs)]

            for i in range(self.num_bucs):
                self.bucs[i].append(i+1)

        elif isinstance(wo_input, str):
            temp_strs = wo_input.split()
            self.num_bucs = len(temp_strs)
            self.bucs = [[] for i in range(self.num_bucs)]

            for i in range(self.num_bucs):
                for s in temp_strs[i].split(","):
                    self.bucs[i].append(int(s))

        elif isinstance(wo_input, list):
            self.num_bucs = len(wo_input)
            self.bucs = wo_input

        else:
            print("Input not recognized")

        # Determine number of objects
        self.n = 0
        for i in range(self.num_bucs):
            self.n = self.n + len(self.bucs[i])

        # Ranking vector of the given object order
        self.ran = np.zeros(self.n, dtype=np.int)

        # Binary vector of the given object order
        self.y = np.zeros(self.n * (self.n - 1))

    # Converts the weak order to a ranking
    def to_ranking(self):

        count = 1

        for b in range(self.num_bucs):
            # Assign proper ranking values, adjusting for number of ties in individual rank position
            for item in self.bucs[b]:
                self.ran[item-1] = count  # item-1 adjusts from object label to corresponding array index
            count = count + len(self.bucs[b])

    # Convert to GKPB solution vector
    def to_binary(self):

        self.to_ranking()

        # Changes ranking vector to (modified) tau-x score matrix (sets 1 if i is preferred/tied with j; 0 if i is not preferred to j)
        if self.num_bucs > 0:
            rankingMat = np.zeros((self.n, self.n), dtype=np.int)
            for i in range(self.n):
                if self.ran[i] != 0:
                    for j in range(self.n - i - 1):
                        j_p = i + j + 1

                        if self.ran[j_p] != 0:
                            # item i is strictly preferred over item j
                            if self.ran[i] < self.ran[j_p]:
                                rankingMat[i, j_p] = 1
                            # item i and j are tied
                            elif self.ran[i] == self.ran[j_p]:
                                rankingMat[i, j_p] = 1
                                rankingMat[j_p, i] = 1
                            # item j is strictly preferred over item i
                            else:
                                rankingMat[j_p, i] = 1

            # Assigns every other value of y lexicographically; even elements of y are the reverse of the odd elements
            # i.e., y-elements are values for comparisons [1,2],[2,1],[1,3],[3,1],...,[1,n],[n,1],[2,3],[3,2],...,[n-1,n],[n,n-1]
            colIdx = 0
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    self.y[colIdx] = rankingMat[i, j]
                    self.y[colIdx + 1] = rankingMat[j, i]
                    colIdx = colIdx + 2

    def objectSearch(self, obj):
        buc_idx = -1
        for i in range(self.num_bucs):
            if obj in self.bucs[i]:
                buc_idx = i

        return buc_idx

    # Shifts alternatives/buckets within a weak order
    # Receives alternatives to move (set I), location of I in the weak order (bucIdx), and the steps parameters (q)
    def move(self, I, q):

        valid_input = True
        # Check inputs are valid
        # Check if all elements of I are in the same bucket
        same_bucket = False
        bucket_idxs = [0]*len(I)
        for i in range(len(I)):
            bucket_idxs[i] = self.objectSearch(I[i])

        if len(bucket_idxs) > 0:
            same_bucket = all(elem == bucket_idxs[0] for elem in bucket_idxs)

        if same_bucket:
            bucIdx = bucket_idxs[0]
        else:
            print("Error: all elements in I does not belong to the same bucket")
            valid_input = False

        # If move is outside index boundaries, flag invalid input
        if bucIdx+q <= -1 or bucIdx+q >= self.num_bucs:
            print("Error: bucket indices exceeded")
            valid_input = False

        if not valid_input:
            print("No moves applied; try again")

        # Input is valid
        else:
            # If q is a whole integer, items in I are moved to an existing bucket
            if isinstance(q, int):
                self.bucs[bucIdx + q] += I

                # Remove elements of I from the bucket
                temp = []
                for i in self.bucs[bucIdx]:
                    if i not in I:
                        temp.append(i)
                self.bucs[bucIdx] = temp

            # If q is not a whole integer, items in I are moved to a new bucket
            else:
                position = int(math.ceil(bucIdx + q))
                self.bucs.insert(position, I)
                if q > 0:
                    temp = []
                    for i in self.bucs[bucIdx]:
                        if i not in I:
                            temp.append(i)
                    self.bucs[bucIdx] = temp
                else:
                    temp = []
                    for i in self.bucs[bucIdx+1]:
                        if i not in I:
                            temp.append(i)
                    self.bucs[bucIdx+1] = temp

        temp = []
        for i in range(len(self.bucs)):
            if len(self.bucs[i]) != 0:
                temp.append(sorted(self.bucs[i]))

        self.bucs = temp
        self.num_bucs = len(self.bucs)


# Merge and Reverse algorithm for CPT1(Type 1 Construction Procedure)
def MR_T1(order_0, I0):

    X_MR = []                   # Characteristic vector matrix
    object_orders = []          # Weak orders

    S1 = WeakOrder(order_0)
    p = S1.num_bucs
    order_0 = copy.deepcopy(S1.bucs)
    S1.to_binary()
    y_0 = copy.deepcopy(S1.y)
    object_orders.append(order_0)
    X_MR.append(y_0)

    for j in range(1, p):
        I1 = S1.bucs[0]
        for k in range(1, p - j + 1):
            S1.move(I1, 1)
            S1.to_binary()
            order_1 = copy.deepcopy(S1.bucs)
            y_1 = copy.deepcopy(S1.y)
            object_orders.append(order_1)
            X_MR.append(y_1)

            S1.move(I1, 0.5)
            S1.to_binary()
            order_2 = copy.deepcopy(S1.bucs)
            y_2 = copy.deepcopy(S1.y)
            object_orders.append(order_2)
            X_MR.append(y_2)

        # Optional outer step; moves items in I0 to first bucket
        S1.move(I0, j-p)

    return S1, object_orders, X_MR


# CPT1 algorithm; Inputs- n: number of objects, N_hat: fixed alternatives
def CPT1(n, N_hat):
    N = [i + 1 for i in range(n)]          # List of all alternatives
    N_hat_comp = []                        # List complement of N_hat

    for i in range(len(N)):
        if i+1 not in N_hat:
            N_hat_comp.append(i+1)

    i1 = N_hat[0]

    # Set beginning weak order
    temp_str = ""

    temp_str = temp_str + str(i1) + "," + str(N_hat_comp[0]) + " "
    for i in range(1, len(N_hat_comp)):
        temp_str = temp_str + str(N_hat_comp[i]) + " "

    order_0 = temp_str

    # Perform M&R: Line 4
    S1, orders, X = MR_T1(order_0, [i1])

    p = S1.num_bucs

    S1.to_binary()
    order_1 = copy.deepcopy(S1.bucs)
    y_1 = copy.deepcopy(S1.y)
    orders.append(order_1)
    X.append(y_1)

    # Line 6
    for j in range(p-1):
        S1.move(S1.bucs[j], 1.5)
        S1.to_binary()
        order_2 = copy.deepcopy(S1.bucs)
        y_2 = copy.deepcopy(S1.y)
        orders.append(order_2)
        X.append(y_2)

    # Line 10
    for j in range(p-1):
        S1.move([i1], -1)
        S1.to_binary()
        order_3 = copy.deepcopy(S1.bucs)
        y_3 = copy.deepcopy(S1.y)
        orders.append(order_3)
        X.append(y_3)

    # Remove the first vector and append it to the end
    orders.insert(len(orders), orders[0])
    X.insert(len(X), X[0])

    orders.pop(0)
    X.pop(0)

    return X, orders


# Creates corresponding labels for the columns
def get_standard_labels(n):

    # Will save variable indices corresponding to each coordinate of the solution vector
    comb_labels = []

    for i in range(n):
        for j in range(i+1, n):
            temp_array = (i+1, j+1)
            comb_labels.append(temp_array)
            temp_array = (j+1, i+1)
            comb_labels.append(temp_array)

    return comb_labels


# Prints the matrices with column labels and row number
def operation_print(matrix, initial_labels):

    col_width = max(len(str(word)) for row in initial_labels for word in row) + 5  # padding
    print("R#\t" + "".join(str(word).ljust(col_width) for word in initial_labels))
    count = 1
    for row in matrix:
        print("R" + str(count) + "\t" + "".join(str(i).center(col_width) for i in row))
        count += 1


# Adds/subtracts rows within a matrix
def row_oper(old_mat, source_idxs, source_scalars, target_idx):

    up_mat = np.zeros((len(old_mat), len(old_mat[0])), dtype=np.int)

    # Deep copy of old matrix
    for i in range(len(old_mat)):
        for j in range(len(old_mat[0])):
            up_mat[i][j] = old_mat[i][j]

    for j in range(len(old_mat[0])):
        sum = 0
        for i in range(len(source_idxs)):
            sum = sum + old_mat[source_idxs[i]][j]*source_scalars[i]

        up_mat[target_idx][j] = sum + old_mat[target_idx][j]

    return up_mat


# Rearrange labels
def permute_varLabels(labels, perm_idxs):
    # Make a deep copy of labels since labels will be overwritten
    temp_labels = copy.deepcopy(labels)

    for i in range(len(labels)):
        labels[i] = temp_labels[perm_idxs[i]]

    return labels


# Rearrange matrix
def permute_solMat(mat, perm_idxs, cols):

    temp_mat = np.zeros((len(mat), len(mat[0])), dtype=np.int)

    # Make a deep copy of mat since it will be overwritten
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            temp_mat[i][j] = mat[i][j]

    # Column permutations
    if cols:
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                mat[i][j] = temp_mat[i][perm_idxs[j]]

    # Row permutations
    else:
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                mat[i][j] = temp_mat[perm_idxs[i]][j]

    return mat


# T1 FDI proof
def T1_Solution_Matrix_Operations(mat, n, N_hat):

    i1 = N_hat[0]

    labels = get_standard_labels(n)

    up_mat = mat

    # Converts floating point matrix to integer matrix
    up_mat = row_oper(up_mat, [0], [0], 1)

    print("\nBefore any operation is performed: ")
    operation_print(up_mat, labels)

    # Subtract previous from current starting at second row from the bottom as the first "current row"
    # and the second row as the last "current row"
    for i in range(n*(n-1)-2, 0, -1):
        up_mat = row_oper(up_mat, [i-1], [-1], i)

    # Subtract last row from first row
    up_mat = row_oper(up_mat, [n*(n-1)-1], [-1], 0)

    print("\nAfter iterative row subtraction is performed (X bar): ")
    operation_print(up_mat, labels)

    # Reorder columns so that those involving j\in N_hat_comp show up first,
    # columns involving i1 and any other alternative show up last
    col_perm_idxs = []
    for i in range(len(labels)):
        if labels[i][0] not in N_hat and labels[i][1] not in N_hat:
            col_perm_idxs.append(i)

    for i in range(len(labels)):
        if labels[i][0] == i1 or labels[i][1] == i1:
            col_perm_idxs.append(i)

    up_mat = permute_solMat(up_mat, col_perm_idxs, 1)               # 1 at the end signals column permutations

    # Change labels to match the corresponding reordering
    labels = permute_varLabels(labels, col_perm_idxs)

    print("\nAfter column permutations (A0 matrix):")
    operation_print(up_mat, labels)

    # Add first n(n-1)/2 even index rows to row n(n-1) (last row)
    for i in range(1, (n-1)*(n-2), 2):
        up_mat = row_oper(up_mat, [i], [1], n * (n - 1) - 1)

    print("\nAfter row addition is performed (A1 matrix): ")
    operation_print(up_mat, labels)

    # Define the labels for the sub-matrices
    labels_B = labels[0:(n-1)*(n-2)]
    labels_C = labels[0:(n-1)*(n-2)]
    labels_D = labels[(n-1)*(n-2):n*(n-1)]
    labels_E = labels[(n-1)*(n-2):n*(n-1)]

    # Define the sub-matrices
    B = up_mat[0:(n-1)*(n-2), 0:(n-1)*(n-2)]
    C = up_mat[(n-1)*(n-2):n*(n-1), 0:(n-1)*(n-2)]
    D = up_mat[0:(n-1)*(n-2), (n-1)*(n-2):n*(n-1)]
    E = up_mat[(n-1)*(n-2):n*(n-1), (n-1)*(n-2):n*(n-1)]

    print("\nB1 matrix:")
    operation_print(B, labels_B)
    print("\nC1 matrix:")
    operation_print(C, labels_C)
    print("\nD1 matrix:")
    operation_print(D, labels_D)
    print("\nE1 matrix:")
    operation_print(E, labels_E)

    # Eliminates entries present in C1
    source_idx1 = (n - 1) * (n - 2) - 1
    source_idx2 = source_idx1 - 1

    for i in range(2, n):
        target_idx = (n - 1) * (n - 2) - 1 + i
        up_mat = row_oper(up_mat, [source_idx1, source_idx2], [1, 1], target_idx)
        source_idx1 = source_idx1 - 2 * (i - 1)
        source_idx2 = source_idx1 - 1

    print("\nAfter turning C into a zero matrix :")

    B = up_mat[0:(n - 1) * (n - 2), 0:(n - 1) * (n - 2)]
    C = up_mat[(n - 1) * (n - 2): n * (n - 1), 0:(n - 1) * (n - 2)]
    D = up_mat[0:(n - 1) * (n - 2), (n - 1) * (n - 2): n * (n - 1)]
    E = up_mat[(n - 1) * (n - 2): n * (n - 1), (n - 1) * (n - 2): n * (n - 1)]

    print("\nB2 matrix:")
    operation_print(B, labels_B)
    print("\nC2 matrix:")
    operation_print(C, labels_C)
    print("\nD2 matrix:")
    operation_print(D, labels_D)
    print("\nE2 matrix:")
    operation_print(E, labels_E)

    # From here on all operations will concentrate only on matrix E

    # Subtract row n from row 2n-2
    E = row_oper(E, [n - 1], [-1], 2 * n - 2 - 1)

    # Add row n+1 to row 2n-2
    E = row_oper(E, [n + 1 - 1], [1], 2 * n - 2 - 1)

    # Add row n-2 to row 2n-2
    E = row_oper(E, [n - 2 - 1], [1], 2 * n - 2 - 1)

    # Add row n+2 to row 2n-2
    E = row_oper(E, [n + 2 - 1], [1], 2 * n - 2 - 1)

    # The following operations are for n>=7
    if n >= 7:
        # Subtract 2*row n-4 from row 2n-2
        E = row_oper(E, [n-4-1], [-2], 2 * n - 2 - 1)

        # Subtract 2*row n+4 from row 2n-2
        E = row_oper(E, [n + 4 - 1], [-2], 2 * n - 2 - 1)

        # The following operations are for n>=8
        if n >= 8:
            # Subtract 5*row n-5 from row 2n-2
            E = row_oper(E, [n - 5 - 1], [-5], 2 * n - 2 - 1)

            # Subtract 5*row n+5 from row 2n-2
            E = row_oper(E, [n + 5 - 1], [-5], 2 * n - 2 - 1)

    # Subtract row (2n-3) from row 2
    E = row_oper(E, [2 * n - 3 - 1], [-1], 1)

    # Add rows 3 to (2n-4) to row (2n-3)
    for i in range(2, 2 * n - 4):
        E = row_oper(E, [i], [1], 2 * n - 3 - 1)

    print("\nE3 matrix:")
    operation_print(E, labels_E)

    print("\ndet(E3) = " + str(round(np.linalg.det(E), 1)))


n = 8
N_hat = [1]
X, orders = CPT1(n, N_hat)

print("The weak orders output by the CPT1 algorithm for n=" + str(n) + " are given below: ")
for i in range(len(orders)):
    tempstr = '{'
    for j in range(len(orders[i])-1):
        tempstr += '{' + str(orders[i][j])[1:-1] + '}, '
    tempstr += '{' + str(orders[i][-1])[1:-1] + '}}'
    print(tempstr)
print('\n')

T1_Solution_Matrix_Operations(X, n, N_hat)





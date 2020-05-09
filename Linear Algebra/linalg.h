#include <iostream>
#include <vector>
#include "math.h"
using namespace std;

class matrix{
    public:
        vector<double> vec;
        int n;
        int m;
        matrix(int n1, int m1){
            n = n1;
            m = m1;
            vec.resize(n*m);
        }
        double get_value(int i, int j){
            return vec[i*m + j];
        }
        void set_value(double val, int i, int j){
            vec[i*m + j] = val;
        }
        vector<double> get_row(int i){
            vector<double> vec1;
            vec1.resize(m);
            for (int j = 0; j < m; j++){
                vec1[j] = get_value(i, j);
            }
            return vec1;
        }
        vector<double> set_row(vector<double> row, int i){
            for (int j = 0; j < m; j++){
                set_value(row[j], i, j);
            }
        }
        vector<double> get_column(int j){
            vector<double> vec1;
            vec1.resize(n);
            for (int i = 0; i < n; i++){
                vec1[i] = get_value(i, j);
            }
            return vec1;
        }
        vector<double> set_column(vector<double> column, int j){
            for(int i = 0; i < n; i++){
                set_value(column[i], i, j);
            }
        }
};

vector<double> copy(vector<double> vec1){
    int n = vec1.size();
    vector<double> vec2;
    vec2.resize(n);
    for (int i = 0; i < n; i++){
        vec2[i] = vec1[i];
    }
    return vec2;
}

matrix copy(matrix mat1){
    int n = mat1.n;
    int m = mat1.m;
    matrix mat2(n, m);
    vector<double> vec2 = copy(mat1.vec);
    mat2.vec = vec2;
    return mat2;
}

vector<double> make_vector(double array[], int n){
    vector<double> vec;
    vec.resize(n);
    for (int i = 0; i < n; i++){
        vec[i] = array[i];
    }
    return vec;
}

matrix make_matrix(vector<double> vec1, int n, int m){
    vector<double> vec2 = copy(vec1);
    matrix mat(n, m);
    mat.vec = vec2;
    return mat;
}

void print(vector<double> vec1){
    vector<double> vec2 = copy(vec1);
    int n = vec2.size();
    for (int i = 0; i < n; i++){
        cout << vec2[i] << endl;
    }
    cout << endl;
}

void print(matrix mat1){
    int n = mat1.n;
    int m = mat1.m;
    matrix mat2(n, m);
    mat2 = copy(mat1);
    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
            cout << mat2.get_value(i, j) << " ";
        }
        cout << endl;
    }
    cout << endl;
}

vector<double> scale_vector(vector<double> vec1, double c){
    int n = vec1.size();
    vector<double> vec2;
    vec2.resize(n);
    for (int i = 0; i < n; i++){
        vec2[i] = c*vec1[i];
    }
    return vec2;
}

vector<double> add_vectors(vector<double> vec1, vector<double> vec2){
    int n = vec1.size();
    vector<double> vec3;
    vec3.resize(n);
    for (int i = 0; i < n; i++){
        vec3[i] = vec1[i] + vec2[i];
    }
    return vec3;
}

double dot(vector<double> vec1, vector<double> vec2){
    int n = vec1.size();
    double total = 0;
    for (int i = 0; i < n; i++){
        total += vec1[i] * vec2[i];
    }
    return total;
}

matrix scale_row(matrix mat1, double c, int i){
    int n = mat1.n;
    int m = mat1.m;
    matrix mat2(n, m);
    mat2 = copy(mat1);
    for (int j = 0; j < m; j++){
        double val = mat2.get_value(i, j);
        mat2.set_value(c*val, i, j);
    }
    return mat2;
}

matrix add_rows(matrix mat1, double c, int i, int j){
    int n = mat1.n;
    int m = mat1.m;
    matrix mat2(n, m);
    mat2 = copy(mat1);
    for (int k = 0; k < m; k++){
        double val1 = mat2.get_value(i, k);
        double val2 = mat2.get_value(j, k);
        mat2.set_value(c*val1 + val2, j, k);
    }
    return mat2;
}

matrix swap_rows(matrix mat1, int i, int j){
    int n = mat1.n;
    int m = mat1.m;
    matrix mat2(n, m);
    mat2 = copy(mat1);
    for (int k = 0; k < m; k++){
        double val1 = mat2.get_value(i, k);
        double val2 = mat2.get_value(j, k);
        mat2.set_value(val1, j, k);
        mat2.set_value(val2, i, k);
    }
    return mat2;
}

matrix get_identity(int n){
    matrix mat(n, n);
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if (i == j){
                mat.set_value(1, i, j);
            }
            else{
                mat.set_value(0, i, j);
            }
        }
    }
    return mat;
}

vector<double> get_zeros(int n){
    vector<double> vec;
    vec.resize(n);
    for(int i = 0; i < n; i++){
        vec[i] = 0;
    }
    return vec;
}

matrix get_zeros(int n, int m){
    matrix mat(n, m);
    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
            mat.set_value(0, i, j);
        }
    }
    return mat;
}

matrix get_transpose(matrix mat1){
    int n = mat1.n;
    int m = mat1.m;
    matrix mat1_copy(n, m);
    mat1_copy = copy(mat1);
    matrix mat2(m, n);
    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
            double val = mat1_copy.get_value(i, j);
            mat2.set_value(val, j, i);
        }
    }
    return mat2;
}

matrix matmul(matrix mat1, matrix mat2){
    int n = mat1.n;
    int p = mat1.m;
    int m = mat2.m;
    matrix mat1_copy(n, p);
    mat1_copy = copy(mat1);
    matrix mat2_copy(p, m);
    mat2_copy = copy(mat2);
    matrix mat3(n, m);
    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
            double total = 0;
            for (int k = 0; k < p; k++){
                total += mat1_copy.get_value(i, k) * mat2_copy.get_value(k, j);
            }
            mat3.set_value(total, i, j);
        }
    }
    return mat3;
}

vector<double> matmul(matrix mat, vector<double> vec1){
    int n = mat.n;
    int m = mat.m;
    matrix mat_copy(n, m);
    mat_copy = copy(mat);
    vector<double> vec1_copy = copy(vec1);
    vector<double> vec2;
    vec2.resize(n);
    for (int i = 0; i < m; i++){
        vector<double> row = mat_copy.get_row(i);
        double total = dot(row, vec1_copy);
        vec2[i] = total;
    }
    return vec2;
}

void get_PALU(matrix A, matrix *P_ptr, matrix *L_ptr, matrix *U_ptr){
    int n = A.n;
    matrix U(n, n);
    matrix P(n, n);
    P = get_identity(n);
    U = copy(A);
    matrix L(n, n);
    L = get_identity(n);
    for (int i = 0; i < n-1; i++){
        double max_value = abs(U.get_value(i, i));
        double max_index = i;
        for(int j = i+1; j < n; j++){
            double val = abs(U.get_value(j, i));
            if (val > max_value){
                max_value = val;
                max_index = j;
            }
            else{}
        }
        U = swap_rows(U, i, max_index);
        P = swap_rows(P, i, max_index);
        for (int j = i+1; j < n; j++){
            double val1 = U.get_value(i, i);
            double val2 = U.get_value(j, i);
            double c = val2 / val1;
            U = add_rows(U, -c, i, j);
            L.set_value(c, j, i);
            U.set_value(0, j, i);
        }
    }
    *P_ptr = P;
    *L_ptr = L;
    *U_ptr = U;
}

vector<double> solve_triangular(matrix mat1, vector<double> b1, string option){
    int n = mat1.n;
    matrix mat1_copy(n, n);
    mat1_copy = copy(mat1);
    vector<double> b1_copy;
    b1_copy = copy(b1);
    vector<double> x;
    x.resize(n);
    if (option == "upper"){
        for (int i = n-1; i >= 0; i--){
            double total = b1_copy[i];
            for (int j = i+1; j < n; j++){
                double val = mat1_copy.get_value(i, j);
                total -= val*x[j];
            }
            x[i] = total/mat1_copy.get_value(i, i);
        }
        return x;
    }
    else{
        for (int i = 0; i < n; i++){
            double total = b1_copy[i];
            for (int j = 0; j < i; j++){
                double val = mat1_copy.get_value(i, j);
                total -= val*x[j];
            }
            x[i] = total/mat1_copy.get_value(i, i);
        }
        return x;
    }
}

vector<double> solve(matrix A, vector<double> b){
    int n = A.n;
    matrix A_copy(n, n);
    A_copy = copy(A);
    vector<double> b_copy = copy(b);
    matrix P(n, n);
    matrix L(n, n);
    matrix U(n, n);
    get_PALU(A_copy, &P, &L, &U);
    vector<double> b2 = matmul(P, b_copy);
    vector<double> c = solve_triangular(L, b2, "lower");
    vector<double> x = solve_triangular(U, c, "upper");
    return x;
}

double get_infinity_norm(vector<double> vec){
    vector<double> vec_copy = copy(vec);
    int n = vec_copy.size();
    double max_abs_value = abs(vec_copy[0]);
    for (int i = 0; i < n; i++){
        double val = abs(vec_copy[i]);
        if (val > max_abs_value){
            max_abs_value = val;
        }
    }
    return max_abs_value;
}

double get_two_norm(vector<double> vec){
    vector<double> vec_copy = copy(vec);
    int n = vec_copy.size();
    return sqrt(dot(vec_copy, vec_copy));
    
}

vector<double> get_projection(vector<double> u, vector<double> a){
    double numerator = dot(u, a);
    double denominator = dot(u, u);
    double c = numerator / denominator;
    vector<double> projection = scale_vector(u, c);
    return projection;
}
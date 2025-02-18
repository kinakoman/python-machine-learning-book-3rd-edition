# カーネルSVM

線形分離が不可能なデータを処理する方法の一つ **カーネル法**である。
カーネル法の基本的な考え方は、射影関数 $\phi$ を用いた組み合わせを高次元空間へ
射影し、線形分離可能にすることである。
すなわち、与えられた特徴量を組み合わせて新たな次元を形成し、その高次元空間で分離を行う。

2次元データセットを新しい3次元空間に変換する射影の1つを下に示す。

$$
\phi(x_1,x_2)=(z_1,z_2,z_3)=(x_1,x_2,x_1^2+x_2^2)
$$

<img src="03_13.png" style="width:50%">

SVMを用いて非線形問題の解を求めるには、射影関数 $\phi$ を用いて訓練データセットをより高い次元の空間に
変換し、この新しい特徴量空間でデータを分類するための線形SVMモデルを訓練する。
このようにすることで、同じ射影関数 $\phi$ を用いて未知のデータを変換し、線形SVMモデルを用いて分類が可能になる。

一方で、新しい特徴量を生成する計算コストが非常に高くなり、より高次元のデータを扱う場合は顕著となる。
**カーネルトリック**計算コストの軽減に用いられる手法であり、カーネル関数を用いる。カーネル関数は以下のように定義される。

$$
\Kappa(\boldsymbol{x}^{(i)},\boldsymbol{x}^{(j)})=\phi(\boldsymbol{x}^{(i)})^T\phi(\boldsymbol{x}^{(j)})
$$

よく利用されているカーネル関数を以下に示す。

- 多項式関数

$$
\Kappa(\boldsymbol{x}^{(i)},\boldsymbol{x}^{(j)})=(\alpha {\boldsymbol{x}^{(i)}}^T \boldsymbol{x}^{(j)}+\beta)^\delta
$$

- ガウスカーネル(RBFカーネル)

$$
\Kappa(\boldsymbol{x}^{(i)},\boldsymbol{x}^{(j)})=\exp\left(-\dfrac{||\boldsymbol{x}^{(i)}-\boldsymbol{x}^{(j)}||^2}{2\sigma}\right)
$$

- シグモイドカーネル

$$
\Kappa(\boldsymbol{x}^{(i)},\boldsymbol{x}^{(j)})=
\dfrac{1}{1+\exp({-\gamma\boldsymbol{x}^{(i)}}^T \boldsymbol{x}^{(j)})}
$$


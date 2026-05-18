import numpy as np

class TestConvergence(unittest.TestCase):
  def test_(self):
    pass
    conv_orders = np.log(errors[1:]/errors[:-1])/np.log(N_list[:-1]/N_list[1:])

  # self.assertEqual((conv_orders == ).all(), f"Convergence order of {scheme} ")

  if __name__ == "__main__":
    unittest.main()

class TestImplicit(unittest.TestCase):
  pass

class TestExplicit(unittest.TestCase):
  pass
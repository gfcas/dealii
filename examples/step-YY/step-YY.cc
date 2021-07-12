#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_relaxation.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/output.h>
#include <deal.II/meshworker/simple.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <mpi.h>

#include <fstream>
#include <iostream>



/**
 * -----------------------------------------------------------------------------
 * With this program we want to solve the heat equation for 1d up to 3d on
 * the domain \Omega = (0,1)^d. The problem states as follows: Find a function
 * u\colon\Omega\to\IR, which satisfies the following equation:
 *                 \ptl_t u = \Delta \u   in (0,T)\times\Omega,
 *      \grad u \cdot \nvec = u^N         on (0,T)\times\ptl\Omega,
 *                u(0,\pkt) = u^0         in \Omega,
 * with the given boundary and initial data u^N and u^0.
 * For this example an exact solution is known:
 *                   u(t,x) = \exp(-d \pi^2 t) \prod_{i=1}^d \sin(\pi x_i).
 * The problem is numerically solved with a FEM of arbitrary polynomial
 * degree and the backward Euler scheme.
 * For the numerical linear algebra the matrixfree framework of deal.II is used.
 * -----------------------------------------------------------------------------
 */
namespace heat
{
  using namespace dealii;

  namespace LA
  {
    using namespace dealii::LinearAlgebra::distributed;
  }


  //--------------------------------------------------------------------------//
  // Global variables                                                         //
  //--------------------------------------------------------------------------//


  const unsigned int n_initial_refinements = 2;

  const unsigned int n_time_steps = 100;


  const double T_final = 0.01;

  const double PI = numbers::PI;



  //--------------------------------------------------------------------------//
  // Exact solution                                                           //
  //--------------------------------------------------------------------------//
  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(const double time);

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

    virtual Tensor<1, dim>
    gradient(const Point<dim> & p,
             const unsigned int component = 0) const override;

    template <typename number>
    number value(const Point<dim, number> &p,
                 const unsigned int        component = 0) const;

    template <typename number>
    Tensor<1, dim, number> gradient(const Point<dim, number> &p,
                                    const unsigned int component = 0) const;
  };



  template <int dim>
  ExactSolution<dim>::ExactSolution(const double time)
    : Function<dim>(1, time)
  {}



  template <int dim>
  double ExactSolution<dim>::value(const Point<dim> & p,
                                   const unsigned int component) const
  {
    return value<double>(p, component);
  }



  template <int dim>
  Tensor<1, dim>
  ExactSolution<dim>::gradient(const Point<dim> & p,
                               const unsigned int component) const
  {
    return gradient<double>(p, component);
  }



  template <int dim>
  template <typename number>
  number ExactSolution<dim>::value(const Point<dim, number> &p,
                                   const unsigned int        component) const
  {
    AssertIndexRange(component, this->n_components);

    number return_value = std::exp(-dim * PI * PI * this->get_time());

    for (unsigned int d = 0; d < dim; ++d)
      return_value *= std::sin(PI * p(d));

    return return_value;
  }



  template <int dim>
  template <typename number>
  Tensor<1, dim, number>
  ExactSolution<dim>::gradient(const Point<dim, number> &p,
                               const unsigned int        component) const
  {
    AssertIndexRange(component, this->n_components);

    Tensor<1, dim, number> return_value;

    for (unsigned int i = 0; i < dim; ++i)
      {
        return_value[i] = PI * std::cos(PI * p(i));

        for (unsigned int d = 0; d < dim; ++d)
          {
            if (d != i)
              {
                return_value[i] *= std::sin(PI * p(d));
              }
          }
      }

    return_value = std::exp(-dim * PI * PI * this->get_time()) * return_value;

    return return_value;
  }



  //--------------------------------------------------------------------------//
  // MatrixFreeOperator SystemMatrix (v,u) + dt*(grad v, grad u)              //
  //--------------------------------------------------------------------------//
  template <int dim, int fe_degree, typename number>
  class SystemMatrixFreeOperator
    : public MatrixFreeOperators::Base<dim, LA::Vector<number>>
  {
  public:
    using value_type = number;

    using FECellIntegrator =
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number>;

    SystemMatrixFreeOperator();

    void set_time_step(const double time_step_in);

    virtual void compute_diagonal() override;

  private:
    virtual void apply_add(LA::Vector<number> &      dst,
                           const LA::Vector<number> &src) const override;

    void
    local_apply(const MatrixFree<dim, number> &              data,
                LA::Vector<number> &                         dst,
                const LA::Vector<number> &                   src,
                const std::pair<unsigned int, unsigned int> &cell_range) const;

    void local_compute_diagonal(FECellIntegrator &integrator) const;

    double time_step;
  };



  template <int dim, int fe_degree, typename number>
  SystemMatrixFreeOperator<dim, fe_degree, number>::SystemMatrixFreeOperator()
    : MatrixFreeOperators::Base<dim, LA::Vector<number>>()
    , time_step(0.0)
  {}



  template <int dim, int fe_degree, typename number>
  void SystemMatrixFreeOperator<dim, fe_degree, number>::set_time_step(
    const double time_step_in)
  {
    Assert(time_step_in > 1e-12, ExcNotInitialized());

    time_step = time_step_in;
  }



  template <int dim, int fe_degree, typename number>
  void SystemMatrixFreeOperator<dim, fe_degree, number>::local_apply(
    const MatrixFree<dim, number> &              data,
    LA::Vector<number> &                         dst,
    const LA::Vector<number> &                   src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FECellIntegrator phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(src,
                            EvaluationFlags::values |
                              EvaluationFlags::gradients);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const VectorizedArray<number> value = phi.get_value(q);
            const Tensor<1, dim, VectorizedArray<number>> gradient =
              phi.get_gradient(q);

            phi.submit_value(value, q);
            phi.submit_gradient(time_step * gradient, q);
          }

        phi.integrate_scatter(EvaluationFlags::values |
                                EvaluationFlags::gradients,
                              dst);
      }
  }



  template <int dim, int fe_degree, typename number>
  void SystemMatrixFreeOperator<dim, fe_degree, number>::apply_add(
    LA::Vector<number> &      dst,
    const LA::Vector<number> &src) const
  {
    this->data->cell_loop(&SystemMatrixFreeOperator::local_apply,
                          this,
                          dst,
                          src);
  }



  template <int dim, int fe_degree, typename number>
  void SystemMatrixFreeOperator<dim, fe_degree, number>::local_compute_diagonal(
    FECellIntegrator &phi) const
  {
    phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    for (unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        phi.submit_value(phi.get_value(q), q);
        phi.submit_gradient(time_step * phi.get_gradient(q), q);
      }

    phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }



  template <int dim, int fe_degree, typename number>
  void SystemMatrixFreeOperator<dim, fe_degree, number>::compute_diagonal()
  {
    this->inverse_diagonal_entries.reset(
      new DiagonalMatrix<LA::Vector<number>>());
    LA::Vector<number> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);

    MatrixFreeTools::compute_diagonal(
      *this->data,
      inverse_diagonal,
      &SystemMatrixFreeOperator::local_compute_diagonal,
      this);

    for (auto &diagonal_element : inverse_diagonal)
      {
        diagonal_element = (std::abs(diagonal_element) > 1.0e-10) ?
                             (1.0 / diagonal_element) :
                             1.0;
      }
  }



  //--------------------------------------------------------------------------//
  // MatrixFree RightHandSide (v,u) + dt*(v,g)                                //
  //--------------------------------------------------------------------------//
  template <int dim, int fe_degree>
  class RightHandSideMatrixFreeOperator : public Subscriptor
  {
  public:
    RightHandSideMatrixFreeOperator();

    void clear();

    void initialize(std::shared_ptr<const MatrixFree<dim, double>> data);

    void set_time_step(const double time_step_in);

    void set_time(const double time_in);

    void apply(LA::Vector<double> &dst, const LA::Vector<double> &src) const;

  private:
    void local_apply_cell(
      const MatrixFree<dim, double> &              data,
      LA::Vector<double> &                         dst,
      const LA::Vector<double> &                   src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

    void local_apply_face(
      const MatrixFree<dim, double> &              data,
      LA::Vector<double> &                         dst,
      const LA::Vector<double> &                   src,
      const std::pair<unsigned int, unsigned int> &face_range) const;

    void local_apply_boundary(
      const MatrixFree<dim, double> &              data,
      LA::Vector<double> &                         dst,
      const LA::Vector<double> &                   src,
      const std::pair<unsigned int, unsigned int> &boundary_range) const;


    double time_step;

    std::shared_ptr<const MatrixFree<dim, double>> data;

    ExactSolution<dim> exact_solution;
  };



  template <int dim, int fe_degree>
  RightHandSideMatrixFreeOperator<dim,
                                  fe_degree>::RightHandSideMatrixFreeOperator()
    : time_step(0.0)
    , exact_solution(0.0)
  {}



  template <int dim, int fe_degree>
  void RightHandSideMatrixFreeOperator<dim, fe_degree>::initialize(
    std::shared_ptr<const MatrixFree<dim, double>> data_in)
  {
    Assert(data_in, ExcNotInitialized());

    data = data_in;
  }



  template <int dim, int fe_degree>
  void RightHandSideMatrixFreeOperator<dim, fe_degree>::clear()
  {
    data.reset();
  }



  template <int dim, int fe_degree>
  void RightHandSideMatrixFreeOperator<dim, fe_degree>::set_time_step(
    const double time_step_in)
  {
    Assert(time_step_in > 1e-12, ExcNotInitialized());

    time_step = time_step_in;
  }



  template <int dim, int fe_degree>
  void RightHandSideMatrixFreeOperator<dim, fe_degree>::set_time(
    const double time_in)
  {
    Assert(time_in > 1e-12, ExcNotInitialized());

    exact_solution.set_time(time_in);
  }



  template <int dim, int fe_degree>
  void RightHandSideMatrixFreeOperator<dim, fe_degree>::local_apply_cell(
    const MatrixFree<dim> &                      data,
    LA::Vector<double> &                         dst,
    const LA::Vector<double> &                   src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);

        phi.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            phi.submit_value(phi.get_value(q), q);
          }
        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
  }



  template <int dim, int fe_degree>
  void RightHandSideMatrixFreeOperator<dim, fe_degree>::local_apply_face(
    const MatrixFree<dim> & /*data*/,
    LA::Vector<double> & /*dst*/,
    const LA::Vector<double> & /*src*/,
    const std::pair<unsigned int, unsigned int> & /*face_range*/) const
  {
    // Assert(false, ExcNotImplemented("No inner face integrals!"));
  }



  template <int dim, int fe_degree>
  void RightHandSideMatrixFreeOperator<dim, fe_degree>::local_apply_boundary(
    const MatrixFree<dim> &data,
    LA::Vector<double> &   dst,
    const LA::Vector<double> & /*src*/,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, double> phi_face(data,
                                                                        true);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        // Check if this is the Neumann boundary
        Assert(data.get_boundary_id(face) == 0, ExcInternalError());

        phi_face.reinit(face);

        for (unsigned int q = 0; q < phi_face.n_q_points; ++q)
          {
            const VectorizedArray<double> test_normal_derivative =
              time_step * phi_face.get_normal_vector(q) *
              exact_solution.gradient(phi_face.quadrature_point(q));

            phi_face.submit_value(test_normal_derivative, q);
          }
        phi_face.integrate_scatter(EvaluationFlags::values, dst);
      }
  }



  template <int dim, int fe_degree>
  void RightHandSideMatrixFreeOperator<dim, fe_degree>::apply(
    LA::Vector<double> &      dst,
    const LA::Vector<double> &src) const
  {
    data->loop(
      &RightHandSideMatrixFreeOperator<dim, fe_degree>::local_apply_cell,
      &RightHandSideMatrixFreeOperator<dim, fe_degree>::local_apply_face,
      &RightHandSideMatrixFreeOperator<dim, fe_degree>::local_apply_boundary,
      this,
      dst,
      src,
      true,
      MatrixFree<dim, double>::DataAccessOnFaces::values,
      MatrixFree<dim, double>::DataAccessOnFaces::none);
  }



  //--------------------------------------------------------------------------//
  // Solver class for the heat equation                                       //
  //--------------------------------------------------------------------------//
  template <int dim, int fe_degree>
  class HeatEquationSolver
  {
  public:
    HeatEquationSolver();

    void run();

    void test_matrix_free();

  private:
    void make_discretization();

    void setup_system();

    void reinit_matrixfree_operators();

    void assemble_matrix_cl();

    void assemble_right_hand_side();

    void assemble_right_hand_side_without_loop();

    void assemble_right_hand_side_cl();

    void solve();

    void compute_next_time_step();

    void compute_next_time_step_cl();

    void graphical_output(const unsigned int cycle);

    void compute_error_norms();

    void build_convergence_table();

    //--------------------------------------------------------------------------

    parallel::distributed::Triangulation<dim> triangulation;

    MappingQ1<dim>  mapping;
    FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler;

    AffineConstraints<double> constraints;
    MGConstrainedDoFs         mg_constrained_dofs;

    using SystemMatrixType = SystemMatrixFreeOperator<dim, fe_degree, double>;
    using LevelMatrixType  = SystemMatrixFreeOperator<dim, fe_degree, float>;

    SystemMatrixType system_matrix;

    MGLevelObject<LevelMatrixType> mg_matrices;

    RightHandSideMatrixFreeOperator<dim, fe_degree> rhs_operator;

    LA::Vector<double> system_rhs;
    LA::Vector<double> current_solution;
    LA::Vector<double> previous_solution;

    LA::Vector<double> exact_solution;
    LA::Vector<double> nodal_error;

    double time;
    double time_step;

    double       h_max;
    unsigned int n_dofs;


    double solver_time;


    double Linfty_error;
    double L2_error;
    double H1s_error;
    double H1_error;


    ConvergenceTable   convergence_table;
    ConditionalOStream pcout;
    TimerOutput        computing_timer;
  };



  template <int dim, int fe_degree>
  HeatEquationSolver<dim, fe_degree>::HeatEquationSolver()
    : triangulation(MPI_COMM_WORLD,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<
                      dim>::construct_multigrid_hierarchy)
    , fe(fe_degree)
    , dof_handler(triangulation)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , computing_timer(MPI_COMM_WORLD,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
  {}



  template <int dim, int fe_degree>
  void HeatEquationSolver<dim, fe_degree>::make_discretization()
  {
    TimerOutput::Scope t(computing_timer, "make discretization");

    // space
    GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
    triangulation.refine_global(n_initial_refinements);
    h_max = GridTools::maximal_cell_diameter(triangulation);

    // time
    time      = 0.0;
    time_step = T_final / n_time_steps;
  }



  template <int dim, int fe_degree>
  void HeatEquationSolver<dim, fe_degree>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup system");

    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();
    n_dofs = dof_handler.n_dofs();

    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    // MatrixFree
    system_matrix.clear();
    rhs_operator.clear();
    {
      typename MatrixFree<dim, double>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
        MatrixFree<dim, double>::AdditionalData::none;
      additional_data.mapping_update_flags =
        (update_values | update_gradients | update_JxW_values |
         update_quadrature_points);


      //////////////////////////////////////////////////////////////////////////
      // If this line is activated, the computation of the system matrix fails.
      additional_data.mapping_update_flags_boundary_faces =
        (update_gradients | update_JxW_values | update_quadrature_points |
         update_normal_vectors);
      //////////////////////////////////////////////////////////////////////////



      auto system_mf_storage = std::make_shared<MatrixFree<dim, double>>();
      system_mf_storage->reinit(mapping,
                                dof_handler,
                                constraints,
                                QGauss<1>(fe_degree + 1),
                                additional_data);
      system_matrix.initialize(system_mf_storage);
      rhs_operator.initialize(system_mf_storage);
    }

    system_matrix.initialize_dof_vector(system_rhs);
    system_matrix.initialize_dof_vector(current_solution);
    system_matrix.initialize_dof_vector(previous_solution);


    // Setup MultiGrid framework
    mg_matrices.clear_elements();

    mg_constrained_dofs.clear();
    mg_constrained_dofs.initialize(dof_handler);

    const unsigned int n_levels = triangulation.n_levels();

    mg_matrices.resize(0, n_levels - 1);

    for (unsigned int level = 0; level < n_levels; ++level)
      {
        IndexSet locally_relevant_level_dofs;
        DoFTools::extract_locally_relevant_level_dofs(
          dof_handler, level, locally_relevant_level_dofs);

        AffineConstraints<double> level_constraints;
        level_constraints.reinit(locally_relevant_level_dofs);
        level_constraints.close();

        typename MatrixFree<dim, float>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme =
          MatrixFree<dim, float>::AdditionalData::none;
        additional_data.mapping_update_flags =
          (update_values | update_gradients | update_JxW_values |
           update_quadrature_points);
        additional_data.mg_level = level;
        auto mg_mf_storage_level = std::make_shared<MatrixFree<dim, float>>();
        mg_mf_storage_level->reinit(mapping,
                                    dof_handler,
                                    level_constraints,
                                    QGauss<1>(fe_degree + 1),
                                    additional_data);
        mg_matrices[level].initialize(mg_mf_storage_level,
                                      mg_constrained_dofs,
                                      level);
      }


    // Interpolate initial condition
    VectorTools::interpolate(dof_handler,
                             ExactSolution<dim>(0.0),
                             current_solution);
  }



  template <int dim, int fe_degree>
  void HeatEquationSolver<dim, fe_degree>::reinit_matrixfree_operators()
  {
    TimerOutput::Scope t(computing_timer, "reinit matrixfree operators");

    system_matrix.set_time_step(time_step);
    system_matrix.compute_diagonal();

    for (unsigned int level = 0; level < triangulation.n_global_levels();
         ++level)
      {
        mg_matrices[level].set_time_step(time_step);
        mg_matrices[level].compute_diagonal();
      }

    rhs_operator.set_time_step(time_step);
  }



  template <int dim, int fe_degree>
  void HeatEquationSolver<dim, fe_degree>::assemble_matrix_cl()
  {
    //     TimerOutput::Scope t(computing_timer, "assemble matrix (classic)");
    //
    //     previous_solution.update_ghost_values();
    //
    //     system_matrix = 0.0;
    //
    //     FEValues<dim> fe_values(fe, QGauss<dim>(fe_degree+1), update_values | update_gradients | update_quadrature_points | update_JxW_values);
    //
    //
    //     const unsigned int n_q_points = fe_values.n_quadrature_points;
    //
    //     const unsigned int dofs_per_cell = fe_values.dofs_per_cell;
    //
    //     FullMatrix<double> cell_matrix(dofs_per_cell);
    //
    //     std::vector<types::global_dof_index>
    //     local_dof_indices(dofs_per_cell);
    //
    //
    //     for(const auto &cell : dof_handler.active_cell_iterators())
    //     {
    //       if(cell->is_locally_owned())
    //       {
    //         cell_matrix = 0.0;
    //
    //         {
    //           fe_values.reinit(cell);
    //
    //           for(unsigned int q=0; q<n_q_points; ++q)
    //           {
    //             const double dx = fe_values.JxW(q);
    //
    //             for(unsigned int i=0; i<dofs_per_cell; ++i)
    //             {
    //               const double phi_i = fe_values.shape_value(i,q);
    //               const Tensor<1,dim> grad_phi_i = fe_values.shape_grad(i,q);
    //
    //               for(unsigned int j=0; j<dofs_per_cell; ++j)
    //               {
    //                 const double phi_j = fe_values.shape_value(j,q);
    //                 const Tensor<1,dim> grad_phi_j =
    //                 fe_values.shape_grad(j,q);
    //
    //                 cell_matrix(i,j) += ( phi_i*phi_j +
    //                 time_step*grad_phi_i*grad_phi_j ) * dx;
    //               }
    //             }
    //           }
    //         }
    //
    //         cell->get_dof_indices(local_dof_indices);
    //
    //         constraints.distribute_local_to_global(cell_matrix,
    //         local_dof_indices, system_matrix);
    //
    //       }
    //     }
    //     system_matrix.compress(VectorOperation::add);
  }



  template <int dim, int fe_degree>
  void HeatEquationSolver<dim, fe_degree>::assemble_right_hand_side_cl()
  {
    TimerOutput::Scope t(computing_timer, "assemble right hand side (classic)");

    previous_solution.update_ghost_values();

    system_rhs = 0.0;

    FEValues<dim>     fe_values(fe,
                            QGauss<dim>(fe_degree + 1),
                            update_values | update_quadrature_points |
                              update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe,
                                     QGauss<dim - 1>(fe_degree + 1),
                                     update_values | update_quadrature_points |
                                       update_normal_vectors |
                                       update_JxW_values);

    const unsigned int n_q_points      = fe_values.n_quadrature_points;
    const unsigned int n_face_q_points = fe_face_values.n_quadrature_points;

    const unsigned int dofs_per_cell = fe_values.dofs_per_cell;
    AssertDimension(dofs_per_cell, fe_face_values.dofs_per_cell);

    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const ExactSolution<dim> exact_solution(time);

    std::vector<double>         u_k(n_q_points);
    std::vector<Tensor<1, dim>> boundary_gradients(n_face_q_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell_rhs = 0.0;

            {
              fe_values.reinit(cell);

              fe_values.get_function_values(previous_solution, u_k);

              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  const double dx = fe_values.JxW(q);

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      cell_rhs(i) += fe_values.shape_value(i, q) * u_k[q] * dx;
                    }
                }
            }


            if (cell->at_boundary())
              {
                for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell;
                     ++f)
                  {
                    if (cell->face(f)->at_boundary())
                      {
                        fe_face_values.reinit(cell, f);

                        const std::vector<Tensor<1, dim>> normal_vectors =
                          fe_face_values.get_normal_vectors();
                        exact_solution.gradient_list(
                          fe_face_values.get_quadrature_points(),
                          boundary_gradients);

                        for (unsigned int q = 0; q < n_face_q_points; ++q)
                          {
                            const double dx = fe_face_values.JxW(q);
                            const double rhs_value =
                              boundary_gradients[q] * normal_vectors[q];

                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                              {
                                cell_rhs(i) +=
                                  time_step * fe_face_values.shape_value(i, q) *
                                  rhs_value * dx;
                              }
                          }
                      }
                  }
              }

            cell->get_dof_indices(local_dof_indices);

            constraints.distribute_local_to_global(cell_rhs,
                                                   local_dof_indices,
                                                   system_rhs);
          }
      }

    system_rhs.compress(VectorOperation::add);
  }


  template <int dim, int fe_degree>
  void
  HeatEquationSolver<dim, fe_degree>::assemble_right_hand_side_without_loop()
  {
    TimerOutput::Scope t(computing_timer, "assemble rhs without loop");

    previous_solution.update_ghost_values();

    system_rhs = 0.0;

    const MatrixFree<dim, double> &data = *system_matrix.get_matrix_free();

    FEEvaluation<dim, fe_degree> phi(data);

    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);

        phi.gather_evaluate(previous_solution, true, false);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const VectorizedArray<double> value = phi.get_value(q);

            phi.submit_value(value, q);
          }
        phi.integrate_scatter(true, false, system_rhs);
      }

    FEFaceEvaluation<dim, fe_degree> phi_face(data, true);

    const ExactSolution<dim> exact_solution(time);

    for (unsigned int face = data.n_inner_face_batches();
         face < data.n_inner_face_batches() + data.n_boundary_face_batches();
         ++face)
      {
        // Check if this is a Neumann boundary face
        Assert(data.get_boundary_id(face) == 0, ExcInternalError());

        phi_face.reinit(face);

        for (unsigned int q = 0; q < phi_face.n_q_points; ++q)
          {
            const VectorizedArray<double> test_normal_derivative =
              time_step * phi_face.get_normal_vector(q) *
              exact_solution.gradient(phi_face.quadrature_point(q));

            phi_face.submit_value(test_normal_derivative, q);
          }
        phi_face.integrate_scatter(true, false, system_rhs);
      }

    system_rhs.compress(VectorOperation::add);
  }



  template <int dim, int fe_degree>
  void HeatEquationSolver<dim, fe_degree>::assemble_right_hand_side()
  {
    TimerOutput::Scope t(computing_timer, "assemble right hand side");

    previous_solution.update_ghost_values();

    system_rhs = 0.0;

    rhs_operator.set_time(time);
    rhs_operator.apply(system_rhs, previous_solution);
  }



  template <int dim, int fe_degree>
  void HeatEquationSolver<dim, fe_degree>::solve()
  {
    TimerOutput::Scope t(computing_timer, "solve");

    MGTransferMatrixFree<dim, float> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    typedef PreconditionChebyshev<LevelMatrixType, LA::Vector<float>> Smoother;
    mg::SmootherRelaxation<Smoother, LA::Vector<float>> mg_smoother;
    MGLevelObject<typename Smoother::AdditionalData>    smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels();
         ++level)
      {
        if (level > 0)
          {
            smoother_data[level].smoothing_range     = 15.0;
            smoother_data[level].degree              = 4;
            smoother_data[level].eig_cg_n_iterations = 10;
          }
        else
          {
            smoother_data[0].smoothing_range = 1e-3;
            smoother_data[0].degree          = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
          }
        smoother_data[level].preconditioner =
          mg_matrices[level].get_matrix_diagonal_inverse();
      }
    mg_smoother.initialize(mg_matrices, smoother_data);

    MGCoarseGridApplySmoother<LA::Vector<float>> mg_coarse;
    mg_coarse.initialize(mg_smoother);

    mg::Matrix<LA::Vector<float>> mg_matrix(mg_matrices);
    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>
      mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels();
         ++level)
      {
        mg_interface_matrices[level].initialize(mg_matrices[level]);
      }

    mg::Matrix<LA::Vector<float>> mg_interface(mg_interface_matrices);

    Multigrid<LA::Vector<float>> mg(
      mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dim, LA::Vector<float>, MGTransferMatrixFree<dim, float>>
      preconditioner(dof_handler, mg, mg_transfer);



    SolverControl                solver_control(100, 1e-12);
    SolverCG<LA::Vector<double>> solver(solver_control);

    constraints.set_zero(current_solution);
    solver.solve(system_matrix, current_solution, system_rhs, preconditioner);
    constraints.distribute(current_solution);
  }



  template <int dim, int fe_degree>
  void HeatEquationSolver<dim, fe_degree>::compute_error_norms()
  {
    TimerOutput::Scope t(computing_timer, "compute error norms");

    current_solution.update_ghost_values();

    QGauss<dim> quadrature(fe_degree + 2);

    Vector<float> difference_per_cell(triangulation.n_active_cells());

    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      current_solution,
                                      ExactSolution<dim>(time),
                                      difference_per_cell,
                                      quadrature,
                                      VectorTools::Linfty_norm);
    Linfty_error = VectorTools::compute_global_error(triangulation,
                                                     difference_per_cell,
                                                     VectorTools::Linfty_norm);

    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      current_solution,
                                      ExactSolution<dim>(time),
                                      difference_per_cell,
                                      quadrature,
                                      VectorTools::L2_norm);
    L2_error = VectorTools::compute_global_error(triangulation,
                                                 difference_per_cell,
                                                 VectorTools::L2_norm);

    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      current_solution,
                                      ExactSolution<dim>(time),
                                      difference_per_cell,
                                      quadrature,
                                      VectorTools::H1_seminorm);
    H1s_error = VectorTools::compute_global_error(triangulation,
                                                  difference_per_cell,
                                                  VectorTools::H1_seminorm);

    H1_error = std::sqrt(L2_error * L2_error + H1s_error * H1s_error);


    // Add the errors to the convergence table
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        convergence_table.add_value("n_cells", triangulation.n_active_cells());
        convergence_table.add_value("n_dofs", n_dofs);
        convergence_table.add_value("Linfty", Linfty_error);
        convergence_table.add_value("L2", L2_error);
        convergence_table.add_value("H1s", H1s_error);
        convergence_table.add_value("H1", H1_error);
        convergence_table.add_value("time", solver_time);
      }
  }



  template <int dim, int fe_degree>
  void HeatEquationSolver<dim, fe_degree>::build_convergence_table()
  {
    TimerOutput::Scope t(computing_timer, "build convergence table");

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        convergence_table.evaluate_convergence_rates(
          "Linfty", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates(
          "L2", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates(
          "H1s", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates(
          "H1", ConvergenceTable::reduction_rate_log2);

        convergence_table.set_precision("Linfty", 2);
        convergence_table.set_precision("L2", 2);
        convergence_table.set_precision("H1s", 2);
        convergence_table.set_precision("H1", 2);
        convergence_table.set_precision("time", 2);

        convergence_table.set_scientific("Linfty", true);
        convergence_table.set_scientific("L2", true);
        convergence_table.set_scientific("H1s", true);
        convergence_table.set_scientific("H1", true);


        // Text output of the convergence table
        convergence_table.write_text(std::cout);
      }
  }



  template <int dim, int fe_degree>
  void
  HeatEquationSolver<dim, fe_degree>::graphical_output(const unsigned int cycle)
  {
    TimerOutput::Scope t(computing_timer, "graphical output");

    if (triangulation.n_global_active_cells() > 1e6)
      return;

    current_solution.update_ghost_values();

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(current_solution, "solution");

    Vector<float> subdomain(triangulation.n_active_cells());
    for (auto &i : subdomain)
      {
        i = triangulation.locally_owned_subdomain();
      }
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(mapping, fe_degree);

    DataOutBase::VtkFlags flags;
    flags.compression_level        = DataOutBase::VtkFlags::best_speed;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);
    data_out.write_vtu_with_pvtu_record(
      "./", "solution_" + std::to_string(dim) + "d", cycle, MPI_COMM_WORLD, 3);

    current_solution.zero_out_ghost_values();
  }



  template <int dim, int fe_degree>
  void HeatEquationSolver<dim, fe_degree>::compute_next_time_step()
  {
    time += time_step;

    previous_solution = current_solution;

    assemble_right_hand_side();
    // assemble_right_hand_side_without_loop();

    solve();
  }



  template <int dim, int fe_degree>
  void HeatEquationSolver<dim, fe_degree>::compute_next_time_step_cl()
  {
    time += time_step;

    previous_solution = current_solution;

    assemble_right_hand_side_cl();

    solve();
  }



  template <int dim, int fe_degree>
  void HeatEquationSolver<dim, fe_degree>::test_matrix_free()
  {
    // Setup the problem

    pcout << "Test computation of the right hand side function" << std::endl;

    make_discretization();
    triangulation.refine_global(3);


    setup_system();
    reinit_matrixfree_operators();

    pcout << "No. DoFs: " << dof_handler.n_dofs() << std::endl;
    time += time_step;


    // Compute the right hand side with matrix-free loop

    previous_solution = current_solution;
    Timer time;

    time.restart();
    assemble_right_hand_side();
    time.stop();
    pcout << "MF+Loop: Time for assemble rhs    (CPU/Wall) " << time.cpu_time()
          << "/" << time.wall_time() << " s" << std::endl;


    LA::Vector<double> rhs_mf_with_loop(system_rhs);


    // Compute the right hand side without matrix-free loop

    previous_solution = current_solution;

    time.restart();
    assemble_right_hand_side_without_loop();
    time.stop();
    pcout << "MF: Time for assemble rhs         (CPU/Wall) " << time.cpu_time()
          << "/" << time.wall_time() << " s" << std::endl;


    LA::Vector<double> rhs_mf_without_loop(system_rhs);


    // Compute the right hand side with classic method

    previous_solution = current_solution;

    time.restart();
    assemble_right_hand_side_cl();
    time.stop();
    pcout << "M:  Time for assemble rhs_cl      (CPU/Wall) " << time.cpu_time()
          << "/" << time.wall_time() << " s" << std::endl;

    LA::Vector<double> rhs_cl(system_rhs);



    // Compute the error

    LA::Vector<double> diff_cl_with_loop(rhs_cl);
    LA::Vector<double> diff_cl_without_loop(rhs_cl);
    diff_cl_with_loop.add(-1.0, rhs_mf_with_loop);
    diff_cl_without_loop.add(-1.0, rhs_mf_without_loop);

    const double error_with_loop    = diff_cl_with_loop.linfty_norm();
    const double error_without_loop = diff_cl_without_loop.linfty_norm();

    pcout << "Error with loop =    " << error_with_loop << std::endl;
    pcout << "Error without loop = " << error_without_loop << std::endl;

    AssertThrow(error_without_loop < 1e-12 && error_with_loop < 1e-12,
                ExcInternalError("Error in RHS"));

    pcout << "All computations are equivalent!" << std::endl;
  }



  template <int dim, int fe_degree>
  void HeatEquationSolver<dim, fe_degree>::run()
  {
    {
      const unsigned int n_ranks =
        Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
      const unsigned int n_vect_doubles = VectorizedArray<double>::size();
      const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;

      std::string DAT_header = "START DATE: " + Utilities::System::get_date() +
                               ", TIME: " + Utilities::System::get_time();
      std::string MPI_header = "Running with " + std::to_string(n_ranks) +
                               " MPI process" + (n_ranks > 1 ? "es" : "");
      std::string VEC_header =
        "Vectorization over " + std::to_string(n_vect_doubles) +
        " doubles = " + std::to_string(n_vect_bits) + " bits (" +
        Utilities::System::get_current_vectorization_level() +
        "), VECTORIZATION_LEVEL=" +
        std::to_string(DEAL_II_COMPILER_VECTORIZATION_LEVEL);
      std::string SOL_header = "Finite element space: " + fe.get_name();

      pcout << std::string(80, '=') << std::endl;
      pcout << DAT_header << std::endl;
      pcout << std::string(80, '-') << std::endl;

      pcout << MPI_header << std::endl;
      pcout << VEC_header << std::endl;
      pcout << SOL_header << std::endl;

      pcout << std::string(80, '=') << std::endl;
    }


    for (unsigned int cycle = 1; cycle <= 5; ++cycle)
      {
        pcout << std::string(80, '-') << std::endl;
        pcout << "Cycle " << cycle << std::endl;
        pcout << std::string(80, '-') << std::endl;


        if (cycle == 1)
          {
            pcout << "Make discretization..." << std::endl;
            make_discretization();
          }
        else
          {
            pcout << "Refine grid and reset time..." << std::endl;
            triangulation.refine_global();

            h_max = GridTools::maximal_cell_diameter(triangulation);

            // time
            time = 0.0;
          }


        Timer timer;


        // Setup system
        pcout << "Setup system..." << std::endl;
        setup_system();

        pcout << "   Triangulation:  " << triangulation.n_global_active_cells()
              << " cells" << std::endl;
        pcout << "   DoFHandler:     " << dof_handler.n_dofs() << " DoFs"
              << std::endl;
        pcout << "   No. time steps: " << n_time_steps << std::endl;
        pcout << std::endl;


        // Reinit matrixfree operators
        pcout << "Reinit matrixfree operators..." << std::endl;
        reinit_matrixfree_operators();


        // Print initial condition
        //       pcout << "Generate inital output..." << std::endl;
        //       graphical_output(0);


        // Start time integration
        pcout << "Start time integration..." << std::endl;
        for (unsigned int time_step_number = 1;
             time_step_number <= n_time_steps;
             ++time_step_number)
          {
            // pcout << "Advancing to time_step " << time_step_number <<
            // std::endl;
            compute_next_time_step();

            // compute_next_time_step_cl();

            // graphical_output(time_step_number);
          }
        timer.stop();

        solver_time = timer.wall_time();
        pcout << "Time for solve (CPU/Wall) " << timer.cpu_time() << "/"
              << timer.wall_time() << " s" << std::endl;
        pcout << std::endl;


        // Compute errors
        pcout << "Compute errors..." << std::endl;
        compute_error_norms();

        pcout << "   Linfty error:   " << Linfty_error << std::endl;
        pcout << "   L2 error:       " << L2_error << std::endl;
        pcout << "   H1s error:      " << H1s_error << std::endl;
        pcout << "   H1 error:       " << H1s_error << std::endl;


        // Graphical output
        pcout << "Generate graphical output..." << std::endl;
        graphical_output(cycle);



        pcout << std::endl;
      }

    // Build convergence table
    pcout << std::string(80, '-') << std::endl;
    build_convergence_table();
    pcout << std::endl;


    {
      pcout << std::string(80, '=') << std::endl;
      pcout << "END DATE: " << Utilities::System::get_date()
            << ", TIME: " << Utilities::System::get_time() << std::endl;
      pcout << std::string(80, '=') << std::endl;
    }
  }
} // namespace heat



int main(int argc, char *argv[])
{
  try
    {
      using namespace heat;

      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);


      // Bug in cell_loop with activated boundary_update_flags
      {
        HeatEquationSolver<2, 1> he;

        he.run();
      }


      // Bug in rhs
      {
        HeatEquationSolver<2, 1> he;

        he.test_matrix_free();
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}

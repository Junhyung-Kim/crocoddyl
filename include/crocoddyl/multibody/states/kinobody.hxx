///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include <pinocchio/algorithm/joint-configuration.hpp>

namespace crocoddyl
{

  template <typename Scalar>
  StateKinodynamicTpl<Scalar>::StateKinodynamicTpl(boost::shared_ptr<PinocchioModel> model)
      : Base(model->nq + model->nv, 2 * model->nv), pinocchio_(model), x0_(VectorXs::Zero(model->nq + model->nv + 8))
  {

    const std::size_t nq0 = model->joints[1].nq();
    x0_.head(nq_) = pinocchio::neutral(*pinocchio_.get());

    // In a Kinodynamic system, we could define the first joint using Lie groups.
    // The current cases are free-flyer (SE3) and spherical (S03).
    // Instead simple represents any joint that can model within the Euclidean manifold.
    // The rest of joints use Euclidean algebra. We use this fact for computing Jdiff.

    // Define internally the limits of the first joint

    lb_.head(nq0) = -3.14 * VectorXs::Ones(nq0);
    ub_.head(nq0) = 3.14 * VectorXs::Ones(nq0);
    lb_.segment(nq0, nq_ - nq0) = pinocchio_->lowerPositionLimit.tail(nq_ - nq0);
    ub_.segment(nq0, nq_ - nq0) = pinocchio_->upperPositionLimit.tail(nq_ - nq0);
    lb_.segment(nq_, nv_) = -pinocchio_->velocityLimit;
    ub_.segment(nq_, nv_) = pinocchio_->velocityLimit;
    lb_.tail(8).head(3) = -1.0 * VectorXs::Ones(3);
    ub_.tail(8).head(3) = 1.0 * VectorXs::Ones(3);
    lb_.tail(4).head(3) = -1.0 * VectorXs::Ones(3);
    ub_.tail(4).head(3) = 1.0 * VectorXs::Ones(3);
    lb_.tail(5).head(1) = -5.0 * VectorXs::Ones(1);
    ub_.tail(5).head(1) = 5.0 * VectorXs::Ones(1);
    lb_.tail(1) = -5.0 * VectorXs::Ones(1);
    ub_.tail(1) = 5.0 * VectorXs::Ones(1);
    Base::update_has_limits();
  }

  template <typename Scalar>
  StateKinodynamicTpl<Scalar>::StateKinodynamicTpl() : Base(), x0_(VectorXs::Zero(0)) {}

  template <typename Scalar>
  StateKinodynamicTpl<Scalar>::~StateKinodynamicTpl() {}

  template <typename Scalar>
  typename MathBaseTpl<Scalar>::VectorXs StateKinodynamicTpl<Scalar>::zero() const
  {
    return x0_;
  }

  template <typename Scalar>
  typename MathBaseTpl<Scalar>::VectorXs StateKinodynamicTpl<Scalar>::rand() const
  {
    VectorXs xrand = VectorXs::Random(nx_);
    xrand.head(nq_) = pinocchio::randomConfiguration(*pinocchio_.get());
    return xrand;
  }

  template <typename Scalar>
  void StateKinodynamicTpl<Scalar>::diff(const Eigen::Ref<const VectorXs> &x0, const Eigen::Ref<const VectorXs> &x1,
                                         Eigen::Ref<VectorXs> dxout) const
  {
    // std::cout << "diff " << x0.tail(4).transpose() << std::endl;
    // std::cout << "diffx " << x1.tail(4).transpose() << std::endl;

    if (static_cast<std::size_t>(x0.size()) != nx_ + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(x1.size()) != nx_ + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x1 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(dxout.size()) != ndx_)
    {
      throw_pretty("Invalid argument: "
                   << "dxout has wrong dimension (it should be " + std::to_string(ndx_) + ")");
    }
    pinocchio::difference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), dxout.head(nv_));
    dxout.segment(nq_, nv_) = x1.segment(nq_, nv_) - x0.segment(nq_, nv_);
    dxout.tail(8) = x1.tail(8) - x0.tail(8);
  }

  template <typename Scalar>
  void StateKinodynamicTpl<Scalar>::diff1(const Eigen::Ref<const VectorXs> &x0, const Eigen::Ref<const VectorXs> &x1,
                                          Eigen::Ref<VectorXs> dxout) const
  {
    // std::cout << "diff " << x0.tail(4).transpose() << std::endl;
    // std::cout << "diffx " << x1.tail(4).transpose() << std::endl;
    if (static_cast<std::size_t>(x0.size()) != nx_ + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(x1.size()) != nx_ + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x1 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(dxout.size()) != 2)
    {
      throw_pretty("Invalid argument: "
                   << "dxout has wrong dimension (it should be " + std::to_string(ndx_) + ")");
    }

    dxout.setZero();
    dxout.head(1) = x1.tail(6).head(1) - x0.tail(6).head(1);
    dxout.tail(1) = x1.tail(2).head(1) - x0.tail(2).head(1);
  }

  template <typename Scalar>
  void StateKinodynamicTpl<Scalar>::integrate(const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &dx,
                                              Eigen::Ref<VectorXs> xout) const
  {
    if (static_cast<std::size_t>(x.size()) != nx_ + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    pinocchio::integrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), xout.head(nq_));
    xout.segment(nq_, nv_) = x.segment(nq_, nv_) + dx.segment(nq_ - 1, nv_);
    xout.tail(8) = x.tail(8) + dx.tail(8);
  }

  template <typename Scalar>
  void StateKinodynamicTpl<Scalar>::Jdiff(const Eigen::Ref<const VectorXs> &x0, const Eigen::Ref<const VectorXs> &x1,
                                          Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                                          const Jcomponent firstsecond) const
  {
    assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));
    if (static_cast<std::size_t>(x0.size()) != nx_ + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(x1.size()) != nx_ + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x1 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }

    if (firstsecond == first)
    {
      if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_)
      {
        throw_pretty("Invalid argument: "
                     << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                            ")");
      }

      pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), Jfirst.topLeftCorner(nv_, nv_),
                             pinocchio::ARG0);
      Jfirst.block(nv_, nv_, nv_, nv_).diagonal().array() = (Scalar)-1;
      Jfirst.bottomRightCorner(8, 8).diagonal().array() = (Scalar)-1;
    }
    else if (firstsecond == second)
    {
      if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_)
      {
        throw_pretty("Invalid argument: "
                     << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                            std::to_string(ndx_) + ")");
      }
      pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), Jsecond.topLeftCorner(nv_, nv_),
                             pinocchio::ARG1);
      Jsecond.block(nv_, nv_, nv_, nv_).diagonal().array() = (Scalar)1;
      Jsecond.bottomRightCorner(8, 8).diagonal().array() = (Scalar)1;
    }
    else
    { // computing both
      if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_)
      {
        throw_pretty("Invalid argument: "
                     << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                            ")");
      }
      if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_)
      {
        throw_pretty("Invalid argument: "
                     << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                            std::to_string(ndx_) + ")");
      }
      pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), Jfirst.topLeftCorner(nv_, nv_),
                             pinocchio::ARG0);
      pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), Jsecond.topLeftCorner(nv_, nv_),
                             pinocchio::ARG1);
      Jfirst.block(nv_, nv_, nv_, nv_).diagonal().array() = (Scalar)-1;
      Jsecond.block(nv_, nv_, nv_, nv_).diagonal().array() = (Scalar)1;
      Jfirst.bottomRightCorner(8, 8).diagonal().array() = (Scalar)-1;
      Jsecond.bottomRightCorner(8, 8).diagonal().array() = (Scalar)1;
    }
  }
  template <typename Scalar>
  void StateKinodynamicTpl<Scalar>::Jdiff1(const Eigen::Ref<const VectorXs> &x0, const Eigen::Ref<const VectorXs> &x1,
                                           Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                                           const Jcomponent firstsecond) const
  {
    assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));
    if (static_cast<std::size_t>(x0.size()) != nx_ + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(x1.size()) != nx_ + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x1 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }

    if (firstsecond == first)
    {
      if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_)
      {
        throw_pretty("Invalid argument: "
                     << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                            ")");
      }

      pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), Jfirst.topLeftCorner(nv_, nv_),
                             pinocchio::ARG0);
      Jfirst.block(nv_, nv_, nv_, nv_).diagonal().array() = (Scalar)-1;
      Jfirst.bottomRightCorner(8, 8).diagonal().array() = (Scalar)-1;
    }
    else if (firstsecond == second)
    {

      // pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), Jsecond.topLeftCorner(nv_, nv_),
      // pinocchio::ARG1);
      // Jsecond.block(nv_, nv_, nv_, nv_).diagonal().array() = (Scalar)1;
      // Jsecond.setIdentity();
      Jsecond.setZero();
      Jsecond.bottomRightCorner(2, 6).topLeftCorner(1, 1).diagonal().array() = (Scalar)1;
      Jsecond.bottomRightCorner(2, 2).bottomLeftCorner(1, 1).diagonal().array() = (Scalar)1;
      // Jsecond.bottomRightCorner(2,2).topLefFtCorner(1,1).diagonal().array() = (Scalar)1;
      // Jsecond.bottomRightCorner(6,6).topLeftCorner(1,1).diagonal().array() = (Scalar)1;
    }
    else
    { // computing both
      if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_)
      {
        throw_pretty("Invalid argument: "
                     << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                            ")");
      }
      if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_)
      {
        throw_pretty("Invalid argument: "
                     << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                            std::to_string(ndx_) + ")");
      }
      pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), Jfirst.topLeftCorner(nv_, nv_),
                             pinocchio::ARG0);
      pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), Jsecond.topLeftCorner(nv_, nv_),
                             pinocchio::ARG1);
      Jfirst.block(nv_, nv_, nv_, nv_).diagonal().array() = (Scalar)-1;
      Jsecond.block(nv_, nv_, nv_, nv_).diagonal().array() = (Scalar)1;
      Jfirst.bottomRightCorner(8, 8).diagonal().array() = (Scalar)-1;
      Jsecond.bottomRightCorner(8, 8).diagonal().array() = (Scalar)1;
    }
  }

  template <typename Scalar>
  void StateKinodynamicTpl<Scalar>::Jintegrate(const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &dx,
                                               Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                                               const Jcomponent firstsecond, const AssignmentOp op) const
  {
    assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));
    assert_pretty(is_a_AssignmentOp(op), ("op must be one of the AssignmentOp {settop, addto, rmfrom}"));

    if (firstsecond == first || firstsecond == both)
    {
      if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_)
      {
        throw_pretty("Invalid argument: "
                     << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                            ")");
      }
      switch (op)
      {
      case setto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jfirst.topLeftCorner(nv_, nv_),
                              pinocchio::ARG0, pinocchio::SETTO);
        Jfirst.bottomRightCorner(nv_ + 8, nv_ + 8).diagonal().array() = (Scalar)1;
        break;
      case addto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jfirst.topLeftCorner(nv_, nv_),
                              pinocchio::ARG0, pinocchio::ADDTO);
        Jfirst.bottomRightCorner(nv_ + 8, nv_ + 8).diagonal().array() += (Scalar)1;
        break;
      case rmfrom:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jfirst.topLeftCorner(nv_, nv_),
                              pinocchio::ARG0, pinocchio::RMTO);
        Jfirst.bottomRightCorner(nv_ + 8, nv_ + 8).diagonal().array() -= (Scalar)1;
        break;
      default:
        throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
        break;
      }
    }
    if (firstsecond == second || firstsecond == both)
    {
      if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_)
      {
        throw_pretty("Invalid argument: "
                     << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                            std::to_string(ndx_) + ")");
      }
      switch (op)
      {
      case setto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jsecond.topLeftCorner(nv_, nv_),
                              pinocchio::ARG1, pinocchio::SETTO);
        Jsecond.setZero();
        Jsecond.bottomRightCorner(nv_ + 8, nv_ + 8).diagonal().array() = (Scalar)1;
        break;
      case addto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jsecond.topLeftCorner(nv_, nv_),
                              pinocchio::ARG1, pinocchio::ADDTO);
        Jsecond.setZero();
        Jsecond.bottomRightCorner(nv_ + 8, nv_ + 8).diagonal().array() += (Scalar)1;
        break;
      case rmfrom:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jsecond.topLeftCorner(nv_, nv_),
                              pinocchio::ARG1, pinocchio::RMTO);
        Jsecond.setZero();

        Jsecond.bottomRightCorner(nv_ + 8, nv_ + 8).diagonal().array() -= (Scalar)1;
        break;
      default:
        throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
        break;
      }
    }
  }

  template <typename Scalar>
  void StateKinodynamicTpl<Scalar>::JintegrateTransport(const Eigen::Ref<const VectorXs> &x,
                                                        const Eigen::Ref<const VectorXs> &dx, Eigen::Ref<MatrixXs> Jin,
                                                        const Jcomponent firstsecond) const
  {
    assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));

    switch (firstsecond)
    {
    case first:
      pinocchio::dIntegrateTransport(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jin.topRows(nv_), pinocchio::ARG0);
      break;
    case second:
      pinocchio::dIntegrateTransport(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jin.topRows(nv_), pinocchio::ARG1);
      break;
    default:
      throw_pretty(
          "Invalid argument: firstsecond must be either first or second. both not supported for this operation.");
      break;
    }
  }

  template <typename Scalar>
  const boost::shared_ptr<pinocchio::ModelTpl<Scalar>> &StateKinodynamicTpl<Scalar>::get_pinocchio() const
  {
    return pinocchio_;
  }

} // namespace crocoddyl
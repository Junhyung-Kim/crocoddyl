///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_KINOBODY_STATES_MULTIBODY_HPP_
#define CROCODDYL_KINOBODY_STATES_MULTIBODY_HPP_

#include <pinocchio/multibody/model.hpp>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/state-base.hpp"
namespace crocoddyl {
  template <typename _Scalar>
  class StateKinodynamicTpl : public StateAbstractTpl<_Scalar>
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef StateAbstractTpl<Scalar> Base;
    typedef pinocchio::ModelTpl<Scalar> PinocchioModel;
    typedef typename MathBase::VectorXs VectorXs;
    typedef typename MathBase::MatrixXs MatrixXs;

    /**
     * @brief Initialize the Kinodynamic state
     *
     * @param[in] model  Pinocchio model
     */
    explicit StateKinodynamicTpl(boost::shared_ptr<PinocchioModel> model);
    StateKinodynamicTpl();
    virtual ~StateKinodynamicTpl();

    /**
     * @brief Generate a zero state.
     *
     * Note that the zero configuration is computed using `pinocchio::neutral`.
     */
    virtual VectorXs zero() const;

    /**
     * @brief Generate a random state
     *
     * Note that the random configuration is computed using `pinocchio::random` which satisfies the manifold definition
     * (e.g., the quaterion definition)
     */
    virtual VectorXs rand() const;

    virtual void diff(const Eigen::Ref<const VectorXs> &x0, const Eigen::Ref<const VectorXs> &x1,
                      Eigen::Ref<VectorXs> dxout) const;
    virtual void diff1(const Eigen::Ref<const VectorXs> &x0, const Eigen::Ref<const VectorXs> &x1,
                       Eigen::Ref<VectorXs> dxout) const;
    virtual void integrate(const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &dx,
                           Eigen::Ref<VectorXs> xout) const;
    virtual void Jdiff(const Eigen::Ref<const VectorXs> &, const Eigen::Ref<const VectorXs> &, Eigen::Ref<MatrixXs> Jfirst,
                       Eigen::Ref<MatrixXs> Jsecond, const Jcomponent firstsecond = both) const;
    virtual void Jdiff1(const Eigen::Ref<const VectorXs> &, const Eigen::Ref<const VectorXs> &, Eigen::Ref<MatrixXs> Jfirst,
                        Eigen::Ref<MatrixXs> Jsecond, const Jcomponent firstsecond = both) const;
    virtual void Jintegrate(const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &dx,
                            Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                            const Jcomponent firstsecond = both, const AssignmentOp = setto) const;
    virtual void JintegrateTransport(const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &dx,
                                     Eigen::Ref<MatrixXs> Jin, const Jcomponent firstsecond) const;

    /**
     * @brief Return the Pinocchio model (i.e., model of the rigid body system)
     */
    const boost::shared_ptr<PinocchioModel> &get_pinocchio() const;

  protected:
    using Base::has_limits_;
    using Base::lb_;
    using Base::ndx_;
    using Base::nq_;
    using Base::nv_;
    using Base::nx_;
    using Base::ub_;

  private:
    boost::shared_ptr<PinocchioModel> pinocchio_; //!< Pinocchio model
    VectorXs x0_;                                 //!< Zero state
  };

} // namespace crocoddyl


/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/states/kinobody.hxx"

#endif  // CROCODDYL_MULTIBODY_STATES_MULTIBODY_HPP_

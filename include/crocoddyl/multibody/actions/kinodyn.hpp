///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, CTU, INRIA, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_KINODYN_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_KINODYN_HPP_

#include <stdexcept>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/multibody/states/kinobody.hpp"
#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"

namespace crocoddyl
{
  template <typename _Scalar>
  class DifferentialActionModelKinoDynamicsTpl : public DifferentialActionModelAbstractTpl<_Scalar>
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef DifferentialActionModelAbstractTpl<Scalar> Base;
    typedef DifferentialActionDataKinoDynamicsTpl<Scalar> Data;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef CostModelSumTpl<Scalar> CostModelSum;
    typedef StateKinodynamicTpl<Scalar> StateKinodynamic;
    typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
    typedef DifferentialActionDataAbstractTpl<Scalar> DifferentialActionDataAbstract;
    typedef typename MathBase::VectorXs VectorXs;
    typedef typename MathBase::MatrixXs MatrixXs;

    DifferentialActionModelKinoDynamicsTpl(boost::shared_ptr<StateKinodynamic> state,
                                           boost::shared_ptr<ActuationModelAbstract> actuation,
                                           boost::shared_ptr<CostModelSum> costs);
    virtual ~DifferentialActionModelKinoDynamicsTpl();

    virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                      const Eigen::Ref<const VectorXs> &u);

    virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract> &data,
                      const Eigen::Ref<const VectorXs> &x);

    virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract> &data,
                          const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u);

    virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract> &data,
                          const Eigen::Ref<const VectorXs> &x);

    virtual boost::shared_ptr<DifferentialActionDataAbstract> createData();

    virtual bool checkData(const boost::shared_ptr<DifferentialActionDataAbstract> &data);

    virtual void quasiStatic(const boost::shared_ptr<DifferentialActionDataAbstract> &data, Eigen::Ref<VectorXs> u,
                             const Eigen::Ref<const VectorXs> &x, const std::size_t maxiter = 100,
                             const Scalar tol = Scalar(1e-9));

    const boost::shared_ptr<ActuationModelAbstract> &get_actuation() const;

    const boost::shared_ptr<CostModelSum> &get_costs() const;

    pinocchio::ModelTpl<Scalar> &get_pinocchio() const;

    const VectorXs &get_armature() const;

    void set_armature(const VectorXs &armature);

    virtual void print(std::ostream &os) const;

  protected:
    using Base::nu_;    //!< Control dimension
    using Base::state_; //!< Model of the state

  private:
    boost::shared_ptr<ActuationModelAbstract> actuation_; //!< Actuation model
    boost::shared_ptr<CostModelSum> costs_;               //!< Cost model
    pinocchio::ModelTpl<Scalar> &pinocchio_;              //!< Pinocchio model
    bool without_armature_;                               //!< Indicate if we have defined an armature
    VectorXs armature_;                                   //!< Armature vector
  };

  template <typename _Scalar>
  struct DifferentialActionDataKinoDynamicsTpl : public DifferentialActionDataAbstractTpl<_Scalar>
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef DifferentialActionDataAbstractTpl<Scalar> Base;
    typedef typename MathBase::VectorXs VectorXs;
    typedef typename MathBase::MatrixXs MatrixXs;

    template <template <typename Scalar> class Model>
    explicit DifferentialActionDataKinoDynamicsTpl(Model<Scalar> *const model)
        : Base(model),
          pinocchio(pinocchio::DataTpl<Scalar>(model->get_pinocchio())),
          multibody(&pinocchio, model->get_actuation()->createData()),
          costs(model->get_costs()->createData(&multibody)),
          Minv(model->get_state()->get_nv(), model->get_state()->get_nv()),
          u_drift(model->get_nu()),
          dtau_dx(model->get_nu(), model->get_state()->get_ndx()),
          tmp_xstatic(model->get_state()->get_nx())
    {
      costs->shareMemory(this);
      Minv.setZero();
      u_drift.setZero();
      dtau_dx.setZero();
      tmp_xstatic.setZero();
    }

    pinocchio::DataTpl<Scalar> pinocchio;
    DataCollectorActMultibodyTpl<Scalar> multibody;
    boost::shared_ptr<CostDataSumTpl<Scalar>> costs;
    MatrixXs Minv;
    VectorXs u_drift;
    MatrixXs dtau_dx;
    VectorXs tmp_xstatic;

    using Base::cost;
    using Base::Fu;
    using Base::Fx;
    using Base::Lu;
    using Base::Luu;
    using Base::Lx;
    using Base::Lxu;
    using Base::Lxx;
    using Base::r;
    using Base::xout;
    using Base::xout2;
    // using Base::dhg;
  };

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include <crocoddyl/multibody/actions/kinodyn.hxx>

#endif  // CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN_HPP_

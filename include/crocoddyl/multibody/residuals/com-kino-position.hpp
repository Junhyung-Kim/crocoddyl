///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_COM_KINOPOSITION_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_COM_KINOPOSITION_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/states/kinobody.hpp"
#include "crocoddyl/multibody/residuals/com-position.hpp"


namespace crocoddyl
{
  template <typename _Scalar>
  class ResidualModelCoMKinoPositionTpl : public ResidualModelAbstractTpl<_Scalar>
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualModelAbstractTpl<Scalar> Base;
    typedef ResidualDataCoMPositionTpl<Scalar> Data;
    typedef StateKinodynamicTpl<Scalar> StateKinodynamic;
    typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef typename MathBase::Vector3s Vector3s;
    typedef typename MathBase::VectorXs VectorXs;

    /**
     * @brief Initialize the CoM position residual model
     *
     * @param[in] state  State of the multibody system
     * @param[in] cref   Reference CoM position
     * @param[in] nu     Dimension of the control vector
     */
    ResidualModelCoMKinoPositionTpl(boost::shared_ptr<StateKinodynamic> state, const std::size_t nu);

    /**
     * @brief Initialize the CoM position residual model
     *
     * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
     *
     * @param[in] state  State of the multibody system
     * @param[in] cref   Reference CoM position
     */
    ResidualModelCoMKinoPositionTpl(boost::shared_ptr<StateKinodynamic> state);
    virtual ~ResidualModelCoMKinoPositionTpl();

    /**
     * @brief Compute the CoM position residual
     *
     * @param[in] data  CoM position residual data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
    virtual void calc(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                      const Eigen::Ref<const VectorXs> &u);

    /**
     * @brief Compute the derivatives of the CoM position residual
     *
     * @param[in] data  CoM position residual data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
    virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                          const Eigen::Ref<const VectorXs> &u);
    virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract *const data);
  /**
   * @brief Return the CoM position reference
   */
  const Vector3s& get_reference() const;

  /**
   * @brief Modify the CoM position reference
   */
  void set_reference(const Vector3s& cref);

    /**
     * @brief Print relevant information of the com-position residual
     *
     * @param[out] os  Output stream object
     */
    virtual void print(std::ostream &os) const;

  protected:
    using Base::nu_;
    using Base::state_;
    using Base::u_dependent_;
    using Base::unone_;
    using Base::v_dependent_;

  private:
    Vector3s cref_;                                                          //!< Reference CoM position
    boost::shared_ptr<typename StateKinodynamic::PinocchioModel> pin_model_; //!< Pinocchio model
  };
} // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/com-kino-position.hxx"

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_COM_POSITION_HPP_

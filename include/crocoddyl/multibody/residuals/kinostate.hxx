///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/kinostate.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl
{

  template <typename Scalar>
  ResidualFlyStateTpl<Scalar>::ResidualFlyStateTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                   const VectorXs &xref, const std::size_t nu)
      : Base(state, 4, nu, false, false, false, false, true), xref_(xref)
  {
    if (static_cast<std::size_t>(xref_.size()) != state_->get_nx() + 11)
    {
      throw_pretty("Invalid argument: "
                   << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
  }

  template <typename Scalar>
  ResidualFlyStateTpl<Scalar>::ResidualFlyStateTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                   const VectorXs &xref)
      : Base(state, 4, false, false, false, false, true), xref_(xref)
  {
    if (static_cast<std::size_t>(xref_.size()) != state_->get_nx() + 11)
    {
      throw_pretty("Invalid argument: "
                   << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
  }

  template <typename Scalar>
  ResidualFlyStateTpl<Scalar>::ResidualFlyStateTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                   const std::size_t nu)
      : Base(state, 4, nu, false, false, false, false, true), xref_(state->zero()) {}

  template <typename Scalar>
  ResidualFlyStateTpl<Scalar>::ResidualFlyStateTpl(boost::shared_ptr<typename Base::StateAbstract> state)
      : Base(state, 4, false, false, false, false, true), xref_(state->zero()) {}

  template <typename Scalar>
  ResidualFlyStateTpl<Scalar>::~ResidualFlyStateTpl() {}

  template <typename Scalar>
  void ResidualFlyStateTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract> &data,
                                         const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u)
  {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx() + 11)
    {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    // state_->diff1(xref_, x, data->r); //diff1
    data->r.setZero();
    data->r.head(1) = x.tail(6+3).head(1) - xref_.tail(6+3).head(1);
    data->r.tail(1) = x.tail(2+3).head(1) - xref_.tail(2+3).head(1);

    data->r.head(2).tail(1) = x.tail(5+3).head(1) - xref_.tail(5+3).head(1);
    data->r.tail(2).head(1) = x.tail(1+3).head(1) - xref_.tail(1+3).head(1);
  }

  template <typename Scalar>
  void ResidualFlyStateTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data,
                                             const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u)
  {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx() + 11)
    {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    // state_->Jdiff1(xref_, x, data->Rx, data->Rx, second);//diff1

    data->Rx.setZero();
    data->Rx.bottomRightCorner(4, 6+3).topLeftCorner(1, 1).diagonal().array() = (Scalar)1;
    data->Rx.bottomRightCorner(2, 2+3).bottomLeftCorner(1, 1).diagonal().array() = (Scalar)1;
  
    data->Rx.bottomRightCorner(3, 5+3).topLeftCorner(1, 1).diagonal().array() = (Scalar)1;
    data->Rx.bottomRightCorner(2, 2+3).topRightCorner(1, 1).diagonal().array() = (Scalar)1;  
  }

  template <typename Scalar>
  void ResidualFlyStateTpl<Scalar>::print(std::ostream &os) const
  {
    os << "ResidualFlyState";
  }

  template <typename Scalar>
  const typename MathBaseTpl<Scalar>::VectorXs &ResidualFlyStateTpl<Scalar>::get_reference() const
  {
    return xref_;
  }

  template <typename Scalar>
  void ResidualFlyStateTpl<Scalar>::set_reference(const VectorXs &reference)
  {
    xref_ = reference;
  }
} // namespace crocoddyl

